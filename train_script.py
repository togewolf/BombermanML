import subprocess
import time

train1 = [
    'python', 'main.py', 'play',
    '--agents', 'Crow_of_Reinforcement', 'Crow_of_Reinforcement', 'rule_based_agent', 'rule_based_agent',
    '--scenario', 'loot-crate',
    '--n-rounds', '50',
    '--train', '1',
    '--no-gui'
]

train2 = [
    'python', 'main.py', 'play',
    '--agents', 'Crow_of_Reinforcement', 'test_copied_agent_ql_tree', 'rule_based_agent', 'coin_collector_agent',
    '--scenario', 'loot-crate',
    '--n-rounds', '50',
    '--train', '1',
    '--no-gui'
]

train3 = [
    'python', 'main.py', 'play',
    '--agents', 'Crow_of_Reinforcement', 'test_copied_agent_ql_tree', 'rule_based_agent', 'coin_collector_agent',
    '--scenario', 'classic',
    '--n-rounds', '50',
    '--train', '1',
    '--no-gui'
]

start = time.time()

subprocess.run(train1)
subprocess.run(train2)
subprocess.run(train3)

end = time.time()

print("Training took " + str (end-start) + "seconds")
# train it a little more using human
