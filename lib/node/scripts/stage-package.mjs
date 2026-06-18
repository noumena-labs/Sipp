import {
  copyFile,
  mkdir,
  readFile,
  readdir,
  rm,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const BINARY_NAME = 'sipp_node';
const ARTIFACT_RE = /^sipp_node_(cpu|cuda|metal|vulkan)\.(.+)\.node$/;

const PLATFORM_TARGETS = {
  'darwin-arm64': {
    cpu: ['arm64'],
    label: 'macOS arm64',
    os: ['darwin'],
  },
  'darwin-x64': {
    cpu: ['x64'],
    label: 'macOS x64',
    os: ['darwin'],
  },
  'linux-x64-gnu': {
    cpu: ['x64'],
    label: 'Linux x64 GNU',
    libc: ['glibc'],
    os: ['linux'],
  },
  'linux-x64-musl': {
    cpu: ['x64'],
    label: 'Linux x64 musl',
    libc: ['musl'],
    os: ['linux'],
  },
  'win32-x64-msvc': {
    cpu: ['x64'],
    label: 'Windows x64 MSVC',
    os: ['win32'],
  },
};

const packageDir = fileURLToPath(new URL('..', import.meta.url));
const repoRoot = path.resolve(packageDir, '..', '..');
const sourceArtifactDir = path.join(repoRoot, '.build', 'artifacts', 'node');
const npmArtifactRoot = path.join(repoRoot, '.build', 'artifacts', 'npm');
const wrapperStageDir = path.join(npmArtifactRoot, 'sipp-server');
const packageJsonPath = path.join(packageDir, 'package.json');

const sourcePackage = JSON.parse(await readFile(packageJsonPath, 'utf8'));
const packageName = process.env.SIPP_NODE_PACKAGE_NAME ?? sourcePackage.name;
const publishConfig = publishConfigFromEnv(sourcePackage);

await mkdir(npmArtifactRoot, { recursive: true });
await removePreviousStageDirs();

const nativeArtifacts = await collectNativeArtifacts();
const packageNamesByTriplet = new Map(
  [...nativeArtifacts.keys()].map((triplet) => [
    triplet,
    platformPackageName(packageName, triplet),
  ]),
);

await stageWrapperPackage(packageNamesByTriplet);
await stagePlatformPackages(nativeArtifacts, packageNamesByTriplet);

function publishConfigFromEnv(packageJson) {
  const registry =
    process.env.SIPP_NODE_PACKAGE_REGISTRY ?? packageJson.publishConfig?.registry;
  const access =
    process.env.SIPP_NODE_PACKAGE_ACCESS ?? packageJson.publishConfig?.access;
  if (registry == null && access == null) {
    return undefined;
  }

  return Object.fromEntries(
    Object.entries({ registry, access }).filter(([, value]) => value != null),
  );
}

async function removePreviousStageDirs() {
  await rm(wrapperStageDir, { recursive: true, force: true });
  const entries = await readdir(npmArtifactRoot, { withFileTypes: true });
  await Promise.all(
    entries
      .filter(
        (entry) => entry.isDirectory() && entry.name.startsWith('sipp-server-'),
      )
      .map((entry) =>
        rm(path.join(npmArtifactRoot, entry.name), {
          force: true,
          recursive: true,
        }),
      ),
  );
}

async function collectNativeArtifacts() {
  const groups = new Map();
  const entries = await readdir(sourceArtifactDir);
  for (const fileName of entries.sort()) {
    const match = ARTIFACT_RE.exec(fileName);
    if (match == null) {
      continue;
    }

    const [, , triplet] = match;
    const target = PLATFORM_TARGETS[triplet];
    if (target == null) {
      throw new Error(`Unsupported Node native target triplet: ${triplet}`);
    }

    const artifacts = groups.get(triplet) ?? [];
    artifacts.push({ fileName });
    groups.set(triplet, artifacts);
  }

  if (groups.size === 0) {
    throw new Error(
      `No ${BINARY_NAME}_*.node artifacts found in ${sourceArtifactDir}`,
    );
  }

  return groups;
}

async function stageWrapperPackage(packageNames) {
  await mkdir(wrapperStageDir, { recursive: true });
  await copyWrapperFiles(wrapperStageDir);

  const optionalDependencies = Object.fromEntries(
    [...packageNames.values()]
      .sort()
      .map((name) => [name, sourcePackage.version]),
  );

  const packageJson = stripDevelopmentMetadata({
    ...sourcePackage,
    files: [
      'index.d.ts',
      'gateway-profile.js',
      'LICENSE',
      'README.md',
      'THIRD_PARTY_NOTICES.md',
      'router.d.ts',
      'router.js',
    ],
    name: packageName,
    optionalDependencies,
    publishConfig,
    repository: {
      ...sourcePackage.repository,
      directory: 'lib/node',
    },
  });

  await writePackageJson(wrapperStageDir, packageJson);
  console.log(`Staged ${packageJson.name} wrapper package`);
}

async function stagePlatformPackages(nativeArtifacts, packageNames) {
  for (const [triplet, artifacts] of nativeArtifacts) {
    const packageDirName = `sipp-server-${triplet}`;
    const packageStageDir = path.join(npmArtifactRoot, packageDirName);
    const nativeDir = path.join(packageStageDir, 'native');
    const target = PLATFORM_TARGETS[triplet];
    await mkdir(nativeDir, { recursive: true });
    await copyCommonFiles(packageStageDir);

    for (const artifact of artifacts) {
      await copyFile(
        path.join(sourceArtifactDir, artifact.fileName),
        path.join(nativeDir, artifact.fileName),
      );
    }

    const packageJson = {
      cpu: target.cpu,
      description: `Native ${target.label} binaries for Sipp Node.js server bindings`,
      files: ['native', 'LICENSE', 'README.md', 'THIRD_PARTY_NOTICES.md'],
      keywords: ['ai', 'gguf', 'llm', 'native', 'sipp'],
      license: sourcePackage.license,
      name: packageNames.get(triplet),
      os: target.os,
      publishConfig,
      repository: {
        ...sourcePackage.repository,
        directory: 'lib/node',
      },
      version: sourcePackage.version,
    };
    if (target.libc != null) {
      packageJson.libc = target.libc;
    }

    await writePackageJson(packageStageDir, packageJson);
    console.log(
      `Staged ${packageJson.name} with ${artifacts.length} native artifact(s)`,
    );
  }
}

function platformPackageName(baseName, triplet) {
  return `${baseName}-${triplet}`;
}

function stripDevelopmentMetadata(packageJson) {
  const clone = { ...packageJson };
  delete clone.devDependencies;
  delete clone.napi;
  delete clone.scripts;
  return clone;
}

async function copyWrapperFiles(stageDir) {
  const packageFiles = [
    'gateway-profile.js',
    'index.d.ts',
    'router.d.ts',
    'router.js',
  ];
  for (const fileName of packageFiles) {
    await copyFile(path.join(packageDir, fileName), path.join(stageDir, fileName));
  }
  await copyCommonFiles(stageDir);
}

async function copyCommonFiles(stageDir) {
  await copyFile(
    path.join(packageDir, 'README.md'),
    path.join(stageDir, 'README.md'),
  );
  await copyFile(path.join(repoRoot, 'LICENSE'), path.join(stageDir, 'LICENSE'));
  await copyFile(
    path.join(repoRoot, 'THIRD_PARTY_NOTICES.md'),
    path.join(stageDir, 'THIRD_PARTY_NOTICES.md'),
  );
}

async function writePackageJson(stageDir, packageJson) {
  await writeFile(
    path.join(stageDir, 'package.json'),
    `${JSON.stringify(packageJson, null, 2)}\n`,
  );
}
