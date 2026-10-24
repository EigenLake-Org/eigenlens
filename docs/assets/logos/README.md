# Database logo sources

These are official project/vendor assets used to identify EigenLens's supported adapters. They replace the illustrative symbols in the original pitch deck. All logos and trademarks belong to their respective owners; their inclusion identifies interoperability and does not imply endorsement.

| Local file | Official source | Notes |
| --- | --- | --- |
| `qdrant.svg` | [Qdrant brand resources](https://qdrant.tech/brand-resources/) · [SVG](https://qdrant.tech/img/brand-resources-logos/qdrant-logo-red-black.svg) | Red/black logo, unchanged |
| `pinecone.svg` | [Pinecone newsroom](https://www.pinecone.io/newsroom/) | Inline website logo; its `--text-primary` theme token is resolved to black for standalone rendering. Paths are unchanged. |
| `milvus.svg` | [Milvus artwork](https://github.com/milvus-io/artwork) · [SVG](https://raw.githubusercontent.com/milvus-io/artwork/master/horizontal/color/milvus-horizontal-color.svg) | Horizontal color logo, unchanged; [trademark usage notes](https://github.com/milvus-io/artwork/blob/master/LICENSE.md) |
| `weaviate.svg` | [Weaviate's website repository](https://github.com/weaviate/weaviate-io) · [SVG](https://raw.githubusercontent.com/weaviate/weaviate-io/main/static/img/site/weaviate-logo-horizontal-dark-1.svg) | Official horizontal logo with dark text, unchanged |
| `postgresql.svg` | [PostgreSQL logo resources](https://wiki.postgresql.org/wiki/Logo) · [SVG](https://wiki.postgresql.org/images/a/a4/PostgreSQL_logo.3colors.svg) | Three-color elephant, unchanged; the adjacent “PostgreSQL + pgvector” text is a caption, not a pgvector logo |

Assets were retrieved on 2026-10-24. The composite preserves their aspect ratios and colors. Source SVGs are kept locally so the README does not depend on external image hosting.

## Regenerate the strip

With Node.js and `sharp` available, run from the repository root:

```bash
node scripts/render_database_logos.cjs
```

For a temporary renderer dependency without changing the Python project:

```bash
npm install --prefix /tmp/eigenlens-logo-renderer sharp
NODE_PATH=/tmp/eigenlens-logo-renderer/node_modules node scripts/render_database_logos.cjs
```

This writes `docs/assets/vector-databases.png` from the checked-in SVGs. No network requests occur during rendering.
