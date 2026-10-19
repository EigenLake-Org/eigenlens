# Publish an EigenLens container release

The release workflow builds and tests the image in GHCR, then mirrors that exact multi-platform digest to the public Docker Hub repository:

```text
sourabheigenlake/eigenlake:eigenlens-0.1.0
```

Docker Hub tags use the `eigenlens-` prefix so this repository can contain more than one image component. The private source repository and GHCR package remain available to maintainers.

## Create a version

Update the package version and the default image version in `compose.release.yaml`, commit the intended release to `main`, then create a stable version tag:

```bash
git tag -a v0.1.1 -m 'EigenLens 0.1.1'
git push origin v0.1.1
```

Use a new tag/version for each release. Do not move an existing release tag. The workflow accepts stable tags of the form `vMAJOR.MINOR.PATCH`.

It builds independently on native AMD64 and ARM64 runners. Each architecture must start successfully and pass an authenticated MCP test covering DBSCAN, BERTopic, summaries, examples and all result pages. Only then does the GHCR publish job create its multi-platform version and commit tags. The Docker Hub workflow pulls the published index by immutable digest, checks each native image, pushes it, and verifies anonymous pulls before publishing `eigenlens-VERSION`, `eigenlens-latest` and `eigenlens-sha-<source-commit>` tags. Architecture build tags are intermediate artifacts, not deployment targets. A failed architecture leaves the previous release tags intact.

The workflow creates a GitHub release and attaches `compose.release.yaml`. Maintainers can use that asset; the [worker guide](worker.md#prebuilt-image) also provides a direct public-image setup without GitHub access. Qdrant uses its upstream image and is not republished.

For a non-release build, run `Publish container image` manually against a branch. It publishes a commit tag on each registry. Numbered and latest tags are updated by stable version tags.

To mirror an already tested GHCR release without rebuilding, run `Publish public Docker Hub image` with its full multi-platform `source_digest` and a `release_version`. Leave `release_version` empty for a commit-only mirror. The source digest must belong to this repository's GHCR package. Do not point a released version at different source content.

## Publisher authentication

The GHCR stage uses `GITHUB_TOKEN` with repository content and package permissions. The Docker Hub stage needs an encrypted repository Actions secret named `DOCKERHUB_TOKEN`, containing a Docker Hub access token with push access to `sourabheigenlake/eigenlake`. The publisher username is `sourabheigenlake`. Secret values are never part of the image, repository or runtime environment.

```bash
# gh prompts privately for the token. Do not put it in a command argument.
gh secret set DOCKERHUB_TOKEN --repo EigenLake-Org/eigenlens
```

Actions are pinned to commit SHAs. OCI source/revision labels retain the original image provenance when it is mirrored. Runtime database credentials, MCP tokens, model keys, datasets and model caches stay outside image builds.

The Docker Hub repository is public. End users do not need a Docker Hub or GitHub login to pull images. GitHub source and release assets remain private; the Docker Hub overview provides public deployment instructions.

After publication, verify the manifest:

```bash
docker buildx imagetools inspect sourabheigenlake/eigenlake:eigenlens-0.1.0
docker pull sourabheigenlake/eigenlake:eigenlens-0.1.0
```

The manifest must include `linux/amd64` and `linux/arm64`. The build logs and GitHub release identify the source commit. See the [Docker Hub documentation](https://docs.docker.com/docker-hub/repos/manage/hub-images/push/) for pushing image tags.
