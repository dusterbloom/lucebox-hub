# run attempts/att2_pool16384_big345

## arm A

| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6712 | - | 6712 | - | - | 6706 | - | 6706 | 6 | 0.1 | 47 | 0.0 | 24.6 | 1.4 |
| 2 | 13825 | - | 13825 | - | - | 6706 | - | 6706 | 7119 | 7.8 | 912 | 0.0 | 24.6 | 9.1 |
| 3 | 22236 | under_threshold | 22236 | 13819 | 4668 | 13819 | - | 13819 | 8417 | 9.9 | 849 | 0.0 | 24.5 | 11.1 |
| 4 | 31067 | under_threshold | 31067 | 13819 | 12860 | 13819 | L=22232 C=22232 replay=8835 | 22232 | 8835 | 10.5 | 845 | 0.0 | 23.9 | 12.7 |
| 5 | 39456 | compress (COMPRESSED) | 6691 | 13819 | 20028 | 0 | L=2048 C=-1 replay=6691 | 0 | 6691 | 7.7 | 871 | 4.9 | 24.3 | 13.8 |

- turn 4 resume: live cut 22232 beats server cut 13819 (slot 0)

- turn 1 answer (23 tok): "(A) The beacon's shutdown forced ships to take fifteen-light-year detours, disrupting main hyperspace routes."
- turn 2 answer (23 tok): "(D) A thick, constantly shifting shell of meteorites and satellite fragments surrounds their planet, making escape impossible."
- turn 3 answer (22 tok): "(C) Ravdin had spotted the Hunter fleet just seven light years away, moving directly toward their city."
- turn 4 answer (49 tok): "Jan was sent to Rathole to pick up a boy suffering from the Venus Shadow, a fatal disease that required immediate evacuation to Earth. He had to cross the dangerous Den Hoorn terrain to reach the settlement before the ship's launch deadline."
- turn 5 answer (26 tok): "(C) He was terrified of the confrontation with Torkleson regarding the disastrous sales and dividend reports waiting on his desk."

