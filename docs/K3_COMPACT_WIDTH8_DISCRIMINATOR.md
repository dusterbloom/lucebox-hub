# K3 compact width-eight discriminator

## Outcome

Width-eight compact execution is materially faster on physical GPU1, but it
is not byte-exact over a representative long input fixture.  It is closed for
the exact production lane.

The one-shot registered sequence first established that a reordered full-12
resident payload was byte-identical for the first widths 1, 2, 4 and 8.  It
then reached a 245-row six-slab identical-mask arm:

| arm | teacher | candidate | nominal speedup | exact |
| --- | ---: | ---: | ---: | --- |
| six-slab identical mask, 245 rows | 65.001480 ms | 15.153320 ms | 4.289587x | no |
| full-12 differing masks, 245 rows | 58.478407 ms | 16.421144 ms | 3.561165x | no |

Both used MMVQ only.  The full-12 arm reduced authoritative test weight H2D
from 722,856,960 to 7,139,328 bytes.  Neither timing can be promoted because
byte identity was a hard prerequisite.

## Decision

The initial eight-input MMVQ result was too narrow; exactness depends on the
input/shape envelope.  The simpler identical-mask proposal from the external
0x-alpha review was tested first and falsified rather than assumed.  The
full-12 union continuation was then run separately without retrying the failed
arm and was also falsified.

No product code remains.  The failed patch is retained at
`/tmp/k3-union-mask-width8-failed.patch` with SHA-256
`8e656ade4579a8b76744273c84799bd68ef16927cfca812be579a5c366b8e0bf`.
The exact runtime must use the proven width-two seam or one-row arithmetic;
width four/eight requires an explicit numerical and model-quality gate.
