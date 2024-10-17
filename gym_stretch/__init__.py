from gymnasium.envs.registration import register

register(
    id="gym_stretch/StretchLiftBox-v0",
    entry_point="gym_stretch.env:StretchEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels", "task": "lift_box"},
)