# README demo media

The main README supports animated demos through ordinary Markdown image links. Keep GIFs in this directory so GitHub displays them alongside the repository documentation.

## Included media

| File | Purpose |
| --- | --- |
| `eigenlake-mark.svg` | Original brand mark from eigenlake.dev; [source](logo-source.md) |
| `eigenlake-mark.png` | PNG export of the website's original SVG mark |
| `eigenlens-hero.svg` | Product headline and a saved BANKING77 point cloud |
| `discovery-workflow.svg` | EigenLens alongside an existing agent's retrieval and action tools |
| `rag-vs-eigenlens.gif` | Illustrative two-column chat: typed question, RAG examples, EigenLens groups and a source follow-up |
| `rag-vs-eigenlens.png` | Static comparison with the completed answers |
| `banking77-discovery.gif` | Four-step workflow illustration: saved selection, analysis plan, clusters and evidence |
| `banking77-discovery.png` | Static cluster-result frame from the walkthrough |
| `banking77-clusters.gif` | Rotating PCA view of all 1,236 selected records |
| `banking77-clusters.png` | Static alternative to the rotating view |
| `banking77-topics.gif` | Three topic views with actual phrases and source examples |
| `vector-databases.png` | Official database artwork on a shared background; [sources and regeneration](logos/README.md) |
| `logos/` | Source SVG assets from the vendors and PostgreSQL project |
| `demo/banking77-payments.json` | Saved coordinates, group counts, topic terms, public examples and provenance |

The GIFs animate **saved analysis output**. Their timing does not represent analysis runtime. The fixture is a saved semantic selection for `payment`, not the full 13,069-record corpus. It contains no embedding vectors or database credentials.

The README's RAG comparison uses illustrative answer wording, not captured model responses or a controlled benchmark. Both columns assume the same saved retrieval selection. The comparison GIF animates typing and an example cluster follow-up; it does not call an agent or a database. Its broader, editorial topic names map `Payment pending` to `Pending card payments`, `Payment app` to `Unrecognized card payments`, and `Payment declined` to `Declined card payments`. This changes presentation only: the saved topic phrases, assignments and counts remain unchanged. The source excerpts are actual fixture records.

The separate workflow GIF illustrates that selection and its analysis; its stages do not represent a live database request or job progress. Cluster/topic counts and the displayed representative record are taken from the fixture.

All selected records come from [mteb/banking77](https://huggingface.co/datasets/mteb/banking77), embedded with [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Dataset and model revisions, analysis parameters and library versions are recorded in the fixture. The saved EigenLens analysis produced 18 clusters, 852 unassigned records and topic phrases. The display uses saved PCA coordinates, not input vectors as chart axes.

## Regenerate the visuals

From the repository root, with Python 3.11+, NumPy and Pillow installed:

```bash
pip install numpy Pillow
python scripts/render_readme_media.py
```

This validates the saved counts and generates the four GIFs, three static previews and two SVGs. It does not query a database, download models or rerun clustering. The script uses Arial on macOS or DejaVu Sans on Linux, with Pillow's default font as a fallback. The database logo strip has a [separate renderer](logos/README.md#regenerate-the-strip).

## Add another demo GIF

1. Capture a working example using public or approved demo data. Remove credentials and private records from both the recording and any saved fixture.
2. Export a readable GIF, ideally no wider than 1,280 pixels and around 2 MB or less. Keep the chart prominent and provide a static PNG alternative.
3. Add the GIF and preview here. Link them from the root README:

   ```markdown
   ![Describe the visible analysis and dataset](docs/assets/my-demo.gif)

   [Static preview](docs/assets/my-demo.png)
   ```

4. Caption the dataset, selection scope and what the demo actually shows. Distinguish recorded interaction from animation of saved results. Do not use playback speed as a runtime claim.

Use a collapsed `<details>` section for extra demos so the main README stays easy to scan. SVGs must be self-contained, without scripts, external fonts or remote images.
