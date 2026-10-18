# Publish an EigenLens container release

The `Publish container image` GitHub Actions workflow builds `ghcr.io/eigenlake-org/eigenlens` using the repository's `GITHUB_TOKEN`. No registry password needs to be stored as a repository secret.

## Create a version

Update the package version and the default image version in `compose.release.yaml`, commit the intended release to `main`, then create a stable version tag:

```bash
git tag -a v0.1.1 -m 'EigenLens 0.1.1'
git push origin v0.1.1
```

Use a new tag/version for each release. Do not move an existing release tag. The workflow accepts stable tags of the form `vMAJOR.MINOR.PATCH`.

It builds independently on native AMD64 and ARM64 runners. Each architecture must start successfully and pass an authenticated MCP test covering DBSCAN, BERTopic, summaries, examples and all result pages. Only then does the publish job create the multi-platform version, `latest` and full-commit `sha-<commit>` tags. Architecture build tags are intermediate artifacts, not deployment targets. A failed architecture leaves the previous release tags intact.

The workflow creates a GitHub release and attaches `compose.release.yaml` after publication. The [worker guide](worker.md#prebuilt-image) shows how to download and use that asset. Qdrant uses its upstream image and is not republished.

For a non-release build, run the workflow manually against a branch or commit in GitHub Actions. It publishes only a `sha-<commit>` deployment tag; `latest` and numbered versions are updated only by a stable release tag. Running manually against an existing version tag republishes that release, so use a branch for checks.

## Package access

The workflow needs `contents: read` and `packages: write` for builds, plus `contents: write` in the final job to attach release assets. Actions are pinned to commit SHAs. OCI source and revision labels link the image to this repository and its source commit. Builds receive no runtime database credentials, MCP token or model API keys; data and models stay in runtime volumes.

GitHub creates new GHCR packages as private. To distribute an image publicly, open the organization's package settings and change that package's visibility to public. The source repository's visibility is separate. Otherwise, give intended users package read access and have them authenticate to GHCR. Repository release downloads still require repository access while the repository is private.

After publication, verify the manifest:

```bash
docker buildx imagetools inspect ghcr.io/eigenlake-org/eigenlens:0.1.0
docker pull ghcr.io/eigenlake-org/eigenlens:0.1.0
```

The manifest must include `linux/amd64` and `linux/arm64`. The build logs and GitHub release identify the source commit. See the [GitHub registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) for package permissions and visibility.
