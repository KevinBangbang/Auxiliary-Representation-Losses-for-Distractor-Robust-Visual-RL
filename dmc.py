# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from pathlib import Path
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class VideoBackgroundWrapper(dm_env.Environment):
    """Replaces clean background with video frames (DMC-GB protocol).

    Uses depth-based segmentation to identify background pixels and replaces
    them with frames from natural videos (e.g. Kinetics-400 clips).
    """
    def __init__(self, env, video_dir, seed=0, render_height=84, render_width=84,
                 camera_id=0):
        self._env = env
        self._video_dir = Path(video_dir)
        self._rng = np.random.RandomState(seed)
        self._render_h = render_height
        self._render_w = render_width
        self._camera_id = camera_id
        self._video_files = sorted(
            list(self._video_dir.glob('*.mp4')) +
            list(self._video_dir.glob('*.avi'))
        )
        if len(self._video_files) == 0:
            raise FileNotFoundError(
                f"No video files found in {video_dir}. "
                "Run: python scripts/download_kinetics.py --mode synthetic")
        self._current_frames = None
        self._frame_idx = 0

    def _load_random_video(self):
        """Load a random video and resize frames to render size."""
        import cv2
        video_path = str(self._rng.choice(self._video_files))
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self._render_w, self._render_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            # Fallback: random noise frames
            frames = [self._rng.randint(0, 256,
                      (self._render_h, self._render_w, 3), dtype=np.uint8)
                      for _ in range(100)]
        return frames

    def _get_background_mask(self):
        """Get binary mask of background pixels using depth rendering."""
        depth = self._env.physics.render(
            height=self._render_h, width=self._render_w,
            camera_id=self._camera_id, depth=True)
        # Background has max/infinite depth
        bg_mask = (depth >= depth.max() * 0.99) | (depth == 0)
        return bg_mask

    def _apply_background(self, time_step):
        """Replace background pixels with video frame."""
        obs_dict = time_step.observation
        if not isinstance(obs_dict, dict) or 'pixels' not in obs_dict:
            return time_step

        pixels_obs = obs_dict['pixels'].copy()
        if len(pixels_obs.shape) == 4:
            pixels_obs = pixels_obs[0]

        h, w = pixels_obs.shape[:2]

        if self._current_frames is None:
            self._current_frames = self._load_random_video()
            self._frame_idx = 0

        bg_frame = self._current_frames[self._frame_idx % len(self._current_frames)]
        self._frame_idx += 1

        mask = self._get_background_mask()
        mask_3d = np.stack([mask] * 3, axis=-1)
        pixels_obs = np.where(mask_3d, bg_frame[:h, :w], pixels_obs)

        new_obs = dict(obs_dict)
        new_obs['pixels'] = pixels_obs
        return time_step._replace(observation=new_obs)

    def reset(self):
        time_step = self._env.reset()
        # Load new random video on each episode
        self._current_frames = None
        return self._apply_background(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._apply_background(time_step)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed,
         use_distractors=False, distractor_video_dir=None):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
        # insert video background distractors after pixel rendering
        if use_distractors and distractor_video_dir is not None:
            env = VideoBackgroundWrapper(
                env, distractor_video_dir, seed=seed,
                render_height=84, render_width=84, camera_id=camera_id)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env
