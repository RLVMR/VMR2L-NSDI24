from gym.envs.registration import register
register(
    id="generalizer-v0",
    entry_point="gym_reschdule_combination.envs:VM_generlizer_v0",
)
register(
    id="generalizer-v1",
    entry_point="gym_reschdule_combination.envs:VM_generlizer_v1",
)
register(
    id="generalizer-v2",
    entry_point="gym_reschdule_combination.envs:VM_generlizer_v2",
)
register(
    id="generalizer-v3",
    entry_point="gym_reschdule_combination.envs:VM_generlizer_v3",
)
register(
    id="penalty-v0",
    entry_point="gym_reschdule_combination.envs:VM_penalty_v0",
)
register(
    id="affinity-v0",
    entry_point="gym_reschdule_combination.envs:VM_affinity_v0",
)
register(
    id="graph-v1",
    entry_point="gym_reschdule_combination.envs:VM_graph_v1",
)
register(
    id="graph-v2",
    entry_point="gym_reschdule_combination.envs:VM_graph_v2",
)