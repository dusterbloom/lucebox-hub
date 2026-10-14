# run /mnt/kimi-k3/split-work/pf746_gap_rebased/run

## arm A

| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6712 | - | 6712 | - | - | 0 | L=1 C=-1 replay=6712 | 0 | 6712 | 6.4 | 1047 | 0.0 | 24.6 | 7.4 |
| 2 | 13825 | - | 13825 | - | - | 6706 | - | 6706 | 7119 | 7.8 | 913 | 0.0 | 24.6 | 8.9 |
| 3 | 22236 | under_threshold | 22236 | 13819 | 4668 | 13819 | - | 13819 | 8417 | 9.9 | 851 | 0.0 | 24.5 | 11.0 |
| 4 | 31067 | under_threshold | 31067 | 13819 | 12860 | 13819 | L=22232 C=22232 replay=8835 | 22232 | 8835 | 10.4 | 848 | 0.0 | 23.9 | 12.6 |
| 5 | 39456 | compress (COMPRESSED) | 6691 | 13819 | 20028 | 0 | L=2048 C=-1 replay=6691 | 0 | 6691 | 6.9 | 975 | 4.8 | 24.3 | 12.9 |

- turn 4 resume: live cut 22232 beats server cut 13819 (slot 2)

- turn 1 answer (23 tok): "(A) The beacon's shutdown forced ships to take fifteen-light-year detours, disrupting main hyperspace routes."
- turn 2 answer (23 tok): "(D) A thick, constantly shifting shell of meteorites and satellite fragments surrounds their planet, making escape impossible."
- turn 3 answer (22 tok): "(C) Ravdin had spotted the Hunter fleet just seven light years away, moving directly toward their city."
- turn 4 answer (49 tok): "Jan was sent to Rathole to pick up a boy suffering from the Venus Shadow, a fatal disease that required immediate evacuation to Earth. He had to cross the dangerous Den Hoorn terrain to reach the settlement before the ship's launch deadline."
- turn 5 answer (26 tok): "(C) He was terrified of the confrontation with Torkleson regarding the disastrous sales and dividend reports waiting on his desk."

## arm B

| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6712 | - | 6712 | - | - | 0 | L=1 C=-1 replay=6712 | 0 | 6712 | 6.5 | 1040 | 0.0 | 24.5 | 7.5 |
| 2 | 13825 | - | 13825 | - | - | 6706 | - | 6706 | 7119 | 7.9 | 905 | 0.0 | 24.5 | 9.0 |
| 3 | 22236 | under_threshold | 22236 | 13819 | 4668 | 13819 | - | 13819 | 8417 | 10.0 | 845 | 0.0 | 24.5 | 11.1 |
| 4 | 31067 | under_threshold | 31067 | 22232 | 4478 | 13819 | L=22232 C=22232 replay=8835 | 22232 | 8835 | 10.5 | 844 | 0.0 | 23.8 | 12.7 |
| 5 | 39456 | under_threshold | 39456 | 31063 | 2843 | 13819 | L=31063 C=31063 replay=8393 | 31063 | 8393 | 10.0 | 842 | 0.0 | 23.9 | 12.2 |

- turn 4 resume: live cut 22232 beats server cut 13819 (slot 2)
- turn 5 resume: live cut 31063 beats server cut 13819 (slot 2)

- turn 1 answer (23 tok): "(A) The beacon's shutdown forced ships to take fifteen-light-year detours, disrupting main hyperspace routes."
- turn 2 answer (23 tok): "(D) A thick, constantly shifting shell of meteorites and satellite fragments surrounds their planet, making escape impossible."
- turn 3 answer (22 tok): "(C) Ravdin had spotted the Hunter fleet just seven light years away, moving directly toward their city."
- turn 4 answer (49 tok): "Jan was sent to Rathole to pick up a boy suffering from the Venus Shadow, a fatal disease that required immediate evacuation to Earth. He had to cross the dangerous Den Hoorn terrain to reach the settlement before the ship's launch deadline."
- turn 5 answer (48 tok): "Walter was anxious because the sales and production reports on his desk were the worst in seven years, indicating the company was failing under union management. He feared that Torkleson would use these dismal figures to fire him for incompetence."

## arm C

| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6712 | - | 6712 | - | - | 0 | L=1 C=-1 replay=6712 | 0 | 6712 | 6.5 | 1037 | 0.0 | 24.5 | 7.5 |
| 2 | 13825 | - | 13825 | - | - | 6706 | - | 6706 | 7119 | 7.9 | 903 | 0.0 | 24.5 | 9.0 |
| 3 | 22236 | - | 22236 | - | - | 13819 | - | 13819 | 8417 | 10.0 | 845 | 0.0 | 24.5 | 11.1 |
| 4 | 31067 | - | 31067 | - | - | 13819 | L=22232 C=22232 replay=8835 | 22232 | 8835 | 10.5 | 844 | 0.0 | 23.8 | 12.7 |
| 5 | 39456 | - | 39456 | - | - | 13819 | L=31063 C=31063 replay=8393 | 31063 | 8393 | 10.0 | 842 | 0.0 | 23.8 | 12.1 |

- turn 4 resume: live cut 22232 beats server cut 13819 (slot 2)
- turn 5 resume: live cut 31063 beats server cut 13819 (slot 2)

- turn 1 answer (23 tok): "(A) The beacon's shutdown forced ships to take fifteen-light-year detours, disrupting main hyperspace routes."
- turn 2 answer (23 tok): "(D) A thick, constantly shifting shell of meteorites and satellite fragments surrounds their planet, making escape impossible."
- turn 3 answer (22 tok): "(C) Ravdin had spotted the Hunter fleet just seven light years away, moving directly toward their city."
- turn 4 answer (49 tok): "Jan was sent to Rathole to pick up a boy suffering from the Venus Shadow, a fatal disease that required immediate evacuation to Earth. He had to cross the dangerous Den Hoorn terrain to reach the settlement before the ship's launch deadline."
- turn 5 answer (48 tok): "Walter was anxious because the sales and production reports on his desk were the worst in seven years, indicating the company was failing under union management. He feared that Torkleson would use these dismal figures to fire him for incompetence."

## answer comparison

- A vs B: t1=identical, t2=identical, t3=identical, t4=identical, t5=DIFFERENT
- A vs C: t1=identical, t2=identical, t3=identical, t4=identical, t5=DIFFERENT
- B vs C: t1=identical, t2=identical, t3=identical, t4=identical, t5=identical
