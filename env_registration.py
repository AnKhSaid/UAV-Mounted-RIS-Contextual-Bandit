from gymnasium.envs.registration import register

# Register the environment
register(
    id="UAV-RIS-v0",
    entry_point="my_env:UAVRISEnv",
)