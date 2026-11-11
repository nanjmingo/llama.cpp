const fs = require('fs');
const path = require('path');

module.exports = async ({ github, core }) => {
    const [owner, repo] = process.env.REPOSITORY.split('/');
    if (!owner || !repo) {
        throw new Error(`Invalid repository name: ${process.env.REPOSITORY}`);
    }

    const releaseTag = process.env.RELEASE_TAG;
    if (!releaseTag) {
        throw new Error('RELEASE_TAG is required.');
    }

    const assetPath = path.resolve(process.env.GITHUB_WORKSPACE, process.env.ASSET_PATH);
    if (!fs.existsSync(assetPath) || !fs.statSync(assetPath).isFile()) {
        throw new Error(`Release asset does not exist: ${assetPath}`);
    }

    const assetName = path.basename(assetPath);

    // getReleaseByTag only returns published releases. The workflow creates a
    // draft release first, so locate it through listReleases instead. GitHub
    // includes draft releases in this endpoint for users with push access.
    const releases = await github.paginate(github.rest.repos.listReleases, {
        owner,
        repo,
        per_page: 100,
    });
    const release = releases.find((candidate) => candidate.tag_name === releaseTag);

    if (!release) {
        throw new Error(
            `Release for tag "${releaseTag}" was not found in the repository's release list. ` +
            'Verify that prepare-prerelease created the release and that this job has contents: write permission.'
        );
    }

    const existingAssets = await github.paginate(github.rest.repos.listReleaseAssets, {
        owner,
        repo,
        release_id: release.id,
    });
    const existingAsset = existingAssets.find((asset) => asset.name === assetName);
    if (existingAsset) {
        await github.rest.repos.deleteReleaseAsset({
            owner,
            repo,
            asset_id: existingAsset.id,
        });
    }

    const assetData = fs.readFileSync(assetPath);
    await github.rest.repos.uploadReleaseAsset({
        owner,
        repo,
        release_id: release.id,
        name: assetName,
        data: assetData,
        headers: {
            'content-type': 'application/octet-stream',
            'content-length': assetData.length,
        },
    });

    if (process.env.PUBLISH_RELEASE === 'true' && release.draft) {
        await github.rest.repos.updateRelease({
            owner,
            repo,
            release_id: release.id,
            draft: false,
            prerelease: true,
        });
    }

    core.info(`Uploaded ${assetName} to ${releaseTag}.`);
};
