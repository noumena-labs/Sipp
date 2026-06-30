const command = process.argv[2];

if (command === 'select-release') {
  const result = selectReleaseVersion({
    baseVersion: requiredEnv('BASE_VERSION'),
    remoteReleaseTags: process.env.REMOTE_RELEASE_TAGS ?? '',
    requestedVersion: requiredEnv('REQUESTED_VERSION').trim(),
  });
  console.log([result.version, result.latestReleaseTag].join('|'));
} else if (command === 'select-dev') {
  const result = selectDevVersion({
    packageName: requiredEnv('PACKAGE'),
    remoteDevTags: process.env.REMOTE_DEV_TAGS ?? '',
    remoteReleaseTags: process.env.REMOTE_RELEASE_TAGS ?? '',
    sourceBaseVersion: requiredEnv('SOURCE_BASE_VERSION'),
  });
  console.log([
    result.npmVersion,
    result.pythonVersion,
    result.tag,
    result.latestReleaseTag,
    result.latestDevTag,
  ].join('|'));
} else {
  throw new Error(`Unknown release version command: ${command ?? '(missing)'}`);
}

function selectReleaseVersion({
  baseVersion,
  remoteReleaseTags,
  requestedVersion,
}) {
  const releaseTags = parseStableReleaseTags(remoteReleaseTags);
  const latestReleaseTag = releaseTags.at(-1)?.tag ?? '';

  if (requestedVersion === 'auto' && releaseTags.length > 0) {
    const latest = releaseTags.at(-1);
    const version = `${latest.major}.${latest.minor}.${latest.patch + 1}`;
    console.error(
      `Latest remote Sipp release tag is ${latest.tag}; auto-selected ${version}.`,
    );
    return { latestReleaseTag, version };
  }

  if (requestedVersion === 'auto') {
    requireStableVersion(baseVersion, 'Source version');
    console.error(
      `No remote Sipp release tags found; using source version ${baseVersion}.`,
    );
    return { latestReleaseTag, version: baseVersion };
  }

  requireStableVersion(requestedVersion, 'release_version');
  console.error(`Using manually requested release version ${requestedVersion}.`);
  return { latestReleaseTag, version: requestedVersion };
}

function selectDevVersion({
  packageName,
  remoteDevTags,
  remoteReleaseTags,
  sourceBaseVersion,
}) {
  const releaseTags = parseStableReleaseTags(remoteReleaseTags);
  const latestRelease = releaseTags.at(-1);
  const latestReleaseTag = latestRelease?.tag ?? '';
  const baseVersion =
    latestRelease == null
      ? sourceBaseVersion
      : `${latestRelease.major}.${latestRelease.minor}.${latestRelease.patch}`;
  requireStableVersion(baseVersion, 'Dev base version');

  const tagScope = packageName === 'all' ? '' : `-${packageName}`;
  const devPrefix = `sipp-dev${tagScope}-v${baseVersion}-dev`;
  const existingDevTags = splitTags(remoteDevTags)
    .filter((tag) => tag.startsWith(devPrefix))
    .map((tag) => ({
      number: Number(tag.slice(devPrefix.length)),
      tag,
    }))
    .filter(({ number }) => Number.isInteger(number) && number > 0)
    .sort((left, right) => left.number - right.number);
  const latestDev = existingDevTags.at(-1);
  const dev = latestDev == null ? 1 : latestDev.number + 1;
  const npmVersion = `${baseVersion}-dev${dev}`;
  const pythonVersion = `${baseVersion}.dev${dev}`;
  const tag = `sipp-dev${tagScope}-v${npmVersion}`;

  console.error(
    `Latest stable tag ${latestReleaseTag || 'none'}; base ${baseVersion}; ` +
      `next dev ${dev}; tag ${tag}`,
  );
  return {
    latestDevTag: latestDev?.tag ?? '',
    latestReleaseTag,
    npmVersion,
    pythonVersion,
    tag,
  };
}

function parseStableReleaseTags(tagText) {
  return splitTags(tagText)
    .map((tag) => /^sipp-v(\d+)\.(\d+)\.(\d+)$/.exec(tag))
    .filter((match) => match != null)
    .map((match) => ({
      tag: match[0],
      major: Number(match[1]),
      minor: Number(match[2]),
      patch: Number(match[3]),
    }))
    .sort(
      (left, right) =>
        left.major - right.major ||
        left.minor - right.minor ||
        left.patch - right.patch,
    );
}

function splitTags(tagText) {
  return tagText
    .split(/\r?\n/)
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function requireStableVersion(version, label) {
  if (!/^\d+\.\d+\.\d+$/.test(version)) {
    throw new Error(`${label} must be a stable x.y.z value, got ${version}`);
  }
}

function requiredEnv(name) {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}
