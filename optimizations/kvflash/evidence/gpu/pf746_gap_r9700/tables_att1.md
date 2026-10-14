# run attempts/att1_pool24576_p28000

## arm A

| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6712 | - | 6712 | - | - | 0 | L=1 C=-1 replay=6712 | 0 | 6712 | 6.5 | 1038 | 0.0 | 22.2 | 7.6 |
| 2 | 13264 | - | 13264 | - | - | 6706 | - | 6706 | 6558 | 7.3 | 902 | 0.0 | 22.1 | 8.7 |
| 3 | 20381 | under_threshold | 20381 | 13258 | 3181 | 13258 | - | 13258 | 7123 | 8.9 | 796 | 0.0 | 22.2 | 10.2 |
| 4 | 27456 | under_threshold | 27456 | 20375 | 2241 | 20375 | - | 20375 | 7081 | 9.5 | 743 | 0.0 | 22.2 | 10.9 |
| 5 | 34933 | under_threshold | 34933 | 20375 | 9409 | 20375 | L=27452 C=27452 replay=7481 | 27452 | 7481 | 10.1 | 738 | 0.0 | 22.1 | 11.4 |

- turn 5 resume: live cut 27452 beats server cut 20375 (slot 3)

- turn 1 answer (23 tok): "(A) The beacon's shutdown forced ships to take fifteen-light-year detours, disrupting main hyperspace routes."
- turn 2 answer (27 tok): "(D) His book argued that human mass suggestion holds the physical fate of the earth, a revolutionary idea that cost him his post."
- turn 3 answer (23 tok): "(D) A thick, constantly shifting shell of meteorites and satellite fragments surrounds their planet, making escape impossible."
- turn 4 answer (22 tok): "(C) Paul plans to ask her to marry him after his mission, indicating she is his romantic interest."
- turn 5 answer (23 tok): "(D) The girls were at the zoo specifically to hear the famous actor-hunter Curt George speak to them."

