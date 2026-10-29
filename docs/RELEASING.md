# Releasing

Releases are cut from a git tag. Pushing a tag `vX.Y.Z` triggers
`.github/workflows/release.yml`, which verifies the tag and that its commit is on `main`, runs
the whole CI workflow on that commit, and creates a GitHub Release from the wheel and sdist CI
built and smoke-tested. Optionally it then publishes the same files to PyPI after a reviewer
approves.

## Versioning policy (SemVer)

The version lives in `pyproject.toml` (`project.version`) and follows
[Semantic Versioning](https://semver.org). The public contract is:

- the command line (`child-mortality`: subcommands, flags, exit codes, error format), and
- the output schema (column names, order, and types of the cube: `cut_size`, `cut`, one column
  per dimension, `rows`, `numerator`, `denominator`, `ratio`; row ordering; null semantics), and
- the documented Python API (`child_mortality.ratio_cube`, `child_mortality.cd_ceb_cube`,
  and the `child_mortality.survey` and `child_mortality.cli` modules).

Bumps:

- MAJOR: breaking change to the output schema or row order, removed or renamed CLI flags or
  subcommands, changed exit codes, removed or incompatible public functions, changed numeric
  results for the same input, or dropping a supported Python version.
- MINOR: backwards-compatible features, such as new flags, new subcommands, new optional
  parameters, or new public functions.
- PATCH: backwards-compatible bug fixes, docs, and internal changes.

Before 1.0.0, minor bumps may include breaking changes.

Pre-releases use PEP 440 suffixes, written in the normalized form: `1.0.0rc1`, `1.0.0b1`,
`1.0.0a1`. Tag them `v1.0.0rc1`. The form `v1.0.0-rc.1` is rejected, because uv normalizes it to
`1.0.0rc1` and the tag must equal `v` plus the project version. Pre-releases are marked as
pre-releases on GitHub.

## Release steps

1. Open a version-bump PR. Start from an up-to-date `main`:

   ```sh
   git switch -c release-vX.Y.Z
   uv version --bump patch   # or minor / major; updates pyproject.toml and uv.lock
   uv version --short        # prints the new version
   git commit -am "Release vX.Y.Z"
   ```

   For a pre-release, combine bumps: `uv version --bump minor --bump alpha` (gives
   `0.2.0a1`), then `uv version --bump alpha` for `a2`, or `uv version --bump beta` /
   `--bump rc` to advance the stage. `uv version 1.0.0rc1` sets it explicitly.
2. Merge that PR first, and make sure CI is green on `main`. Never tag an unmerged commit: the
   release workflow fails before building if the tagged commit is not an ancestor of `main`.
3. Pull `main` and confirm the version:

   ```sh
   git switch main && git pull
   uv version --short        # must print X.Y.Z
   ```
4. Tag and push:

   ```sh
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
5. Watch the Release workflow (Actions tab, or `gh run watch`). It fails early if the tag is
   not semver, does not equal `v` plus the project version, or is not on `main`. If PyPI
   publishing is enabled, approve the pending `pypi` deployment when prompted.

The workflow can also be run manually (Actions, Release, Run workflow) with the name of an
existing tag.

## What the workflow does

Four jobs run in order; a failure stops the rest.

1. `verify` (`contents: read`): checks out the tag with full history. Fails if the tag is not
   `vMAJOR.MINOR.PATCH[aN|bN|rcN]`, differs from `v` + `uv version --short`, or points at a commit
   that is not an ancestor of `origin/main`. Nothing is built or installed for a bad tag.
2. `ci` (`contents: read`): calls `ci.yml` as a reusable workflow on the tagged commit. That is
   lint, lockfile, format and type checks, tests on Linux, macOS and Windows, tests at the lowest
   supported polars, then the build job: it builds the sdist and wheel, checks the wheel contents
   (`py.typed` present, no `tests/` or `data/`), installs the wheel in a fresh venv and runs the CLI,
   and uploads `dist/`. A wheel that fails any of this cannot be released.
3. `release` (the only job with `contents: write`): downloads that `dist/` artifact and creates the
   GitHub Release with generated notes and the files attached. Pre-releases are marked as such. It
   runs no project code: no dependency install, no tests, no build.
4. If the repository variable `PUBLISH_TO_PYPI` is `true`: `publish-pypi` waits for approval on the
   `pypi` environment, then publishes the same artifact with `pypa/gh-action-pypi-publish` using
   trusted publishing. Nothing is rebuilt. Until you set the variable, the job is skipped.

The uv version is pinned in `ci.yml` and `release.yml` (`UV_VERSION`; Renovate bumps both).

## Repository settings that back the workflow

Workflow checks run only after a tag exists, so also restrict who can create tags:

- Settings, Rules, Rulesets: add a tag ruleset for the pattern `v*` that restricts creation (and
  blocks deletion and updates) to maintainers. Without it, anyone with write access can push a tag
  and start the workflow (it still cannot release a commit that is not on `main`).
- Settings, Environments, `pypi`: **Required reviewers must be enabled** (at least one person
  other than whoever pushes the tag, where the team size allows). This is what stops a single
  stolen or mistaken tag push from publishing an immutable PyPI release.

## Publishing to PyPI (one-time setup)

The repository is currently private. GitHub Releases work for private repositories, but
publishing to PyPI makes the package, and its source distribution, public. Check that the name
`child-mortality` is still available on PyPI before the first upload. Publishing uses Trusted
Publishing (OIDC), so no API token is stored.

1. Create a PyPI account at https://pypi.org and enable 2FA.
2. Add a pending trusted publisher at https://pypi.org/manage/account/publishing/ with: PyPI
   project name `child-mortality`, owner `evonnec`, repository `child_mortality`, workflow
   `release.yml`, environment `pypi`. The first successful publish creates the project.
3. In the GitHub repo, Settings, Environments, create an environment named `pypi`.
   Add required reviewers to the environment (required, not optional), so each publish waits for
   a manual approval. Also add the `v*` tag ruleset described above.
4. Enable the job:

   ```sh
   gh variable set PUBLISH_TO_PYPI --body true
   ```

5. Make sure project metadata in `pyproject.toml` (description, readme, authors, URLs,
   license) is complete before the first upload.

Optional dry run: register a pending publisher at https://test.pypi.org with the same values,
and publish a pre-release manually with `uv publish --publish-url https://test.pypi.org/legacy/`
from a trusted environment. The workflow itself targets PyPI only.

To stop publishing, delete the variable: `gh variable delete PUBLISH_TO_PYPI`.

## Recovering from a bad tag

If the Release workflow failed, or the tag points at the wrong commit, and nothing reached PyPI:

```sh
git tag -d vX.Y.Z
git push origin :refs/tags/vX.Y.Z
gh release delete vX.Y.Z --yes   # only if a GitHub Release was created
```

Fix the problem on `main` through a PR, then tag again from step 3 above.

## Yanking a bad release

PyPI versions are immutable: a filename can never be re-uploaded, even after deletion. If
`X.Y.Z` reached PyPI with a defect, do not reuse the number. Fix forward with a new patch
version, and yank the bad release: on https://pypi.org/manage/project/child-mortality/releases/
open the release, choose Options, Yank, and give a reason. Yanked releases stay installable
when pinned exactly (`==X.Y.Z`) but are skipped by normal resolution. Mark the GitHub Release
as a pre-release or edit its notes to point at the fix.

## Installing a released wheel

```sh
uv tool install https://github.com/evonnec/child_mortality/releases/download/vX.Y.Z/child_mortality-X.Y.Z-py3-none-any.whl
child-mortality --version
```

Once published to PyPI: `uv tool install child-mortality`.
