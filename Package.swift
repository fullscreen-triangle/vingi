// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Vingi",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "VingiCore",
            targets: ["VingiCore"]
        ),
        .library(
            name: "VingiMacOS",
            targets: ["VingiMacOS"]
        ),
        .library(
            name: "VingiIOS",
            targets: ["VingiIOS"]
        ),
        .library(
            name: "VingiBridge",
            targets: ["VingiBridge"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-crypto.git", from: "3.0.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-async-algorithms.git", from: "1.0.0"),
        .package(url: "https://github.com/kishikawakatsumi/KeychainAccess.git", from: "4.2.0"),
        .package(url: "https://github.com/vapor/vapor.git", from: "4.0.0")
    ],
    targets: [
        // Core Framework
        .target(
            name: "VingiCore",
            dependencies: [
                .product(name: "Crypto", package: "swift-crypto"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Collections", package: "swift-collections"),
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
                .product(name: "KeychainAccess", package: "KeychainAccess")
            ],
            path: "src/core/VingiCore/Sources/VingiCore",
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals"),
                .enableUpcomingFeature("ConciseMagicFile"),
                .enableUpcomingFeature("ForwardTrailingClosures"),
                .enableUpcomingFeature("ImplicitOpenExistentials"),
                .enableUpcomingFeature("StrictConcurrency")
            ]
        ),
        
        // macOS Platform Target
        .target(
            name: "VingiMacOS",
            dependencies: ["VingiCore"],
            path: "src/core/VingiMacOS/Sources/VingiMacOS"
        ),
        
        // iOS Platform Target
        .target(
            name: "VingiIOS",
            dependencies: ["VingiCore"],
            path: "src/core/VingiIOS/Sources/VingiIOS"
        ),
        
        // Python Bridge
        .target(
            name: "VingiBridge",
            dependencies: [
                "VingiCore",
                .product(name: "Vapor", package: "vapor")
            ],
            path: "src/bridge/VingiBridge/Sources/VingiBridge"
        ),
        
        // Test Targets
        .testTarget(
            name: "VingiCoreTests",
            dependencies: ["VingiCore"],
            path: "src/core/VingiCore/Tests/VingiCoreTests"
        ),
        
        .testTarget(
            name: "VingiBridgeTests",
            dependencies: ["VingiBridge"],
            path: "src/bridge/VingiBridge/Tests/VingiBridgeTests"
        )
    ],
    swiftLanguageVersions: [.v5]
)
