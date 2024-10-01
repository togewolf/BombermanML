# Bomberman reinforcement learning project

## Our final agent:

agent_code/Crow_of_Reinforcement



## File naming explanation
For the convenience of the reader, here is an explanation of the file names:
In the beginning, there were four folders: `agent1_rf`, `agent2_qlnn`, `agent3_ga`, and `agent4_neat`, named after
techniques we intended to implement later: Random Forest, q-learning neural network, genetic algorithm ,
and NeuroEvolution of Augmenting Topologies. 
The initial random forest agent was quickly abandoned (although another futile attempt was made later, 
with more advanced features: `agent1_rfV2`), and due to the success of the q-learning neural agent and time constraints,
we never got to the others. `agent2_qlnn` was the first working agent, which was split up into `agent2_qlnn`,
where Ole developed the convolutional agent, and `agent2_qlnnV2` where Thomas continued improving the features
and architecture of the original q-learning neural agent. Those feature and architecture improvements were
transferred to the `agent2_qlnn` agent before the labors on the convolutional network were discontinued. 
Thus began the era of carefully optimizing features, architecture, and training methods in `agent2_qlnnV2`, during which,
occasionally, agents were saved for testing purposes and to check whether changes did indeed enhance the agent:
`agent2_qlnnV2_old_1` up to `agent2_qlnnV2_old_5`. Our efforts culminated in the renaming of the latest `agent2_qlnnV2` 
agent, which shall henceforth be known as the almighty "`Crow of Reinforcement`", which we then submitted for the competition.
