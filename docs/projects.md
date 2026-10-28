# Saved projects and caching

A `ProjectDocument` stores named operation nodes and explicit dependency
edges. Each node contains the same request accepted by the standalone
Python function and CLI. Studio saves this document directly.

Run the bundled graph from the repository root:

```bash
uv run pic2panning validate examples/project.json
uv run pic2panning run examples/project.json
```

The [example](../examples/project.json) prepares the waterfall photo, then
exports its social variants. Its `targets: ["social"]` selects the social
node and its preparation dependency. The separate `film` node is available
when a single movie is wanted. The examples use the public coast and
aurora photos with an original eight-second, 120 BPM soundtrack distributed
under this repository's MIT license.

## Declare files and dependencies

`ProjectRequest` wraps the document with a `workspace`, a JSON report
`output`, optional target identifiers, a cache policy, and `overwrite`.
Relative node paths resolve from that workspace. Absolute paths remain
absolute in the Python and CLI interfaces. Studio confines them to its
selected workspace.

When one node reads another node's declared output, add the producer's
identifier to `depends_on`. An empty target list runs every enabled node;
otherwise only targets and their transitive dependencies execute. Node
order in the document controls display order. Execution follows a stable
topological order.

Before processing starts, the runner rejects cycles, missing dependencies,
duplicate producers, and output aliases to inputs or other outputs. It
also discovers files referenced through depth manifests. Model directories
and private cache storage receive output protection. Project mask nodes
must merge selections into one declared PNG.

The complete graph can include edits, model operations, compositions,
previews, beat analysis, or social packs. A model-generated image becomes
an ordinary image input for a later node. A generated video can become a
video layer. Library code does not substitute another backend after a
failure.

## Reuse intermediates

| `cache` value | Behavior |
| --- | --- |
| `reuse` | Verify cached artifacts and reuse matching work; default |
| `refresh` | Regenerate selected operations and update their cache entries |
| `off` | Execute through temporary staging without persistent cache storage |

Fingerprints include operation settings, actual input bytes, output
formats and topology, processing code, and model identity checks. Files
such as reference images, masks, fonts, local weights, and depth artifacts
participate in invalidation. Changing a destination filename can reuse an
existing result; changing PNG to JPEG changes its processing identity.

Cached files live under `.pic2panning/cache/` in the workspace. Every reuse
checks artifact hashes. A damaged cache raises an error; use `refresh` to
regenerate it explicitly. A workspace lock prevents simultaneous runs
from publishing into the same cache. Its kernel ownership releases when
a worker exits or is canceled.

Republishing existing outputs still requires `overwrite: true`. That flag
permits replacing declared outputs and the report while keeping source
aliases protected. Completed nodes remain reusable after a later node
fails. The report publishes once every selected node finishes.

## Read the execution report

`ProjectResult.outputs` lists the report and produced artifacts. Each
`nodes` entry includes its identifier, requested output paths, `cache_hit`,
elapsed seconds, fingerprint, input hashes, and the canonical operation's
result. Published paths replace private cache locations in that result.

Keep the project JSON with this report when reproducing an export.
`ProjectDocument.revision` tracks editor saves; operation content and
inputs determine cached processing work.

For individual commands using the same assets:

```bash
uv run pic2panning run examples/compose.json
uv run pic2panning run examples/preview.json
uv run pic2panning run examples/beats.json
uv run pic2panning run examples/social-pack.json
```

The standalone composition uses the original public photos directly.
Previews and social stills call the same frame renderer as the movie.
Safe guides produce separate preview images when requested. See the
[tool guide](agent-tools.md) for layer, audio, and export settings.
