import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { resolve, sep } from "node:path";

const archiveName = "SippCore.xcframework.zip";
const managedPaths = [
  "Binary",
  "LICENSE",
  "Package.swift",
  "README.md",
  "SIPP_SOURCE_REVISION",
  "Sources",
  "Tests",
];
const requiredSnapshotPaths = [
  "LICENSE",
  "Package.swift",
  "README.md",
  "SIPP_SOURCE_REVISION",
  "Sources",
  "Tests",
];
const localBinaryTarget =
  /\.binaryTarget\(\s*name:\s*"SippCore",\s*path:\s*"Binary\/SippCore\.xcframework"\s*\)/g;

function stageSwiftDistribution({
  checksum,
  destination,
  distributionReadme,
  license,
  sourcePackage,
  sourceRepository,
  sourceRevision,
  version,
  distributionRepository,
}) {
  requireMatch(
    distributionRepository,
    /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/,
    "SWIFT_DISTRIBUTION_REPOSITORY",
  );
  requireMatch(
    sourceRepository,
    /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/,
    "SIPP_SOURCE_REPOSITORY",
  );
  requireMatch(
    version,
    /^\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?$/,
    "SWIFT_VERSION",
  );
  requireMatch(checksum, /^[0-9a-fA-F]{64}$/, "SWIFT_CHECKSUM");
  requireMatch(sourceRevision, /^[0-9a-f]{40}$/, "SIPP_SOURCE_REVISION");

  const sourceRoot = resolve(sourcePackage);
  const destinationRoot = resolve(destination);
  const manifestPath = resolveManagedPath(sourceRoot, "Package.swift");
  const generatedBindings = resolveManagedPath(
    sourceRoot,
    "Sources/SippCoreBindings/Generated/SippCoreBindings.swift",
  );
  for (const requiredPath of [
    manifestPath,
    generatedBindings,
    resolveManagedPath(sourceRoot, "Sources"),
    resolveManagedPath(sourceRoot, "Tests"),
    resolve(distributionReadme),
    resolve(license),
  ]) {
    if (!existsSync(requiredPath)) {
      throw new Error(
        `Required Swift distribution input is missing: ${requiredPath}`,
      );
    }
  }

  const archiveUrl =
    `https://github.com/${distributionRepository}/releases/download/` +
    `${version}/${archiveName}`;
  const sourceManifest = readFileSync(manifestPath, "utf8");
  const matches = [...sourceManifest.matchAll(localBinaryTarget)];
  if (matches.length !== 1) {
    throw new Error(
      `Expected exactly one local SippCore binary target in ${manifestPath}, ` +
        `found ${matches.length}`,
    );
  }
  const distributionManifest = sourceManifest.replace(
    localBinaryTarget,
    [
      ".binaryTarget(",
      '            name: "SippCore",',
      `            url: "${archiveUrl}",`,
      `            checksum: "${checksum.toLowerCase()}"`,
      "        )",
    ].join("\n"),
  );

  mkdirSync(destinationRoot, { recursive: true });
  for (const relativePath of managedPaths) {
    rmSync(resolveManagedPath(destinationRoot, relativePath), {
      force: true,
      recursive: true,
    });
  }

  cpSync(
    resolveManagedPath(sourceRoot, "Sources"),
    resolveManagedPath(destinationRoot, "Sources"),
    { recursive: true },
  );
  cpSync(
    resolveManagedPath(sourceRoot, "Tests"),
    resolveManagedPath(destinationRoot, "Tests"),
    { recursive: true },
  );
  cpSync(
    resolve(distributionReadme),
    resolveManagedPath(destinationRoot, "README.md"),
  );
  cpSync(resolve(license), resolveManagedPath(destinationRoot, "LICENSE"));
  writeFileSync(
    resolveManagedPath(destinationRoot, "Package.swift"),
    distributionManifest,
  );
  writeFileSync(
    resolveManagedPath(destinationRoot, "SIPP_SOURCE_REVISION"),
    [
      `repository=https://github.com/${sourceRepository}`,
      `revision=${sourceRevision}`,
      `version=${version}`,
      `artifact=${archiveName}`,
      `artifact_sha256=${checksum.toLowerCase()}`,
      "",
    ].join("\n"),
  );

  return {
    archiveName,
    archiveUrl,
    destination: destinationRoot,
  };
}

function syncSwiftDistribution({ destination, source }) {
  const sourceRoot = resolve(source);
  const destinationRoot = resolve(destination);
  if (sourceRoot === destinationRoot) {
    throw new Error("Swift distribution source and destination must differ");
  }

  for (const relativePath of requiredSnapshotPaths) {
    const sourcePath = resolveManagedPath(sourceRoot, relativePath);
    if (!existsSync(sourcePath)) {
      throw new Error(`Swift distribution snapshot is missing: ${sourcePath}`);
    }
  }

  mkdirSync(destinationRoot, { recursive: true });
  for (const relativePath of managedPaths) {
    const sourcePath = resolveManagedPath(sourceRoot, relativePath);
    const destinationPath = resolveManagedPath(destinationRoot, relativePath);
    rmSync(destinationPath, { force: true, recursive: true });
    if (existsSync(sourcePath)) {
      cpSync(sourcePath, destinationPath, { recursive: true });
    }
  }
}

function resolveManagedPath(root, relativePath) {
  const resolvedRoot = resolve(root);
  const path = resolve(resolvedRoot, relativePath);
  if (path !== resolvedRoot && !path.startsWith(`${resolvedRoot}${sep}`)) {
    throw new Error(`Managed path escapes ${resolvedRoot}: ${relativePath}`);
  }
  return path;
}

function requireMatch(value, pattern, name) {
  if (!pattern.test(value)) {
    throw new Error(`${name} has an invalid value: ${value}`);
  }
}

function requiredEnvironment(name) {
  const value = process.env[name];
  if (value == null || value === "") {
    throw new Error(`${name} is required`);
  }
  return value;
}

function main() {
  const command = process.argv[2];
  if (command === "sync") {
    syncSwiftDistribution({
      destination: requiredEnvironment("SWIFT_DISTRIBUTION_DIR"),
      source: requiredEnvironment("SWIFT_DISTRIBUTION_SOURCE"),
    });
    return;
  }

  if (command === "stage") {
    const result = stageSwiftDistribution({
      checksum: requiredEnvironment("SWIFT_CHECKSUM"),
      destination: requiredEnvironment("SWIFT_DISTRIBUTION_DIR"),
      distributionReadme:
        process.env.SWIFT_DISTRIBUTION_README ??
        "lib/swift/Distribution/README.md",
      distributionRepository: requiredEnvironment(
        "SWIFT_DISTRIBUTION_REPOSITORY",
      ),
      license: process.env.SIPP_LICENSE ?? "LICENSE",
      sourcePackage: requiredEnvironment("SWIFT_PACKAGE_SOURCE"),
      sourceRepository:
        process.env.SIPP_SOURCE_REPOSITORY ?? "noumena-labs/Sipp",
      sourceRevision: requiredEnvironment("SIPP_SOURCE_REVISION"),
      version: requiredEnvironment("SWIFT_VERSION"),
    });
    console.error(
      `Staged Swift ${requiredEnvironment("SWIFT_VERSION")} in ` +
        `${result.destination} for ${result.archiveUrl}`,
    );
    return;
  }

  throw new Error(
    `Unknown Swift distribution command: ${command ?? "(missing)"}`,
  );
}

main();
