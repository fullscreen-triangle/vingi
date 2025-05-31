import Foundation
import Combine
import CryptoKit
import Logging
import KeychainAccess

/// Main entry point for the Vingi Personal AI Assistant framework
@MainActor
public final class VingiCore: ObservableObject {
    
    // MARK: - Singleton
    public static let shared = VingiCore()
    
    // MARK: - Published Properties
    @Published public private(set) var isInitialized = false
    @Published public private(set) var isRunning = false
    @Published public private(set) var currentContext: UserContext?
    @Published public private(set) var systemStatus: SystemStatus = .idle
    
    // MARK: - Core Components
    private var configuration: VingiConfiguration?
    private var contextManager: ContextManager?
    private var intelligenceEngine: IntelligenceEngine?
    private var automationEngine: AutomationEngine?
    private var securityManager: SecurityManager?
    private var logger: Logger
    
    // MARK: - Cancellables
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    private init() {
        self.logger = Logger(label: "com.vingi.core")
        logger.info("VingiCore initializing...")
    }
    
    /// Initialize the Vingi framework with configuration
    /// - Parameter config: Configuration for the framework
    /// - Throws: VingiError if initialization fails
    public func initialize(config: VingiConfiguration) async throws {
        guard !isInitialized else {
            logger.warning("VingiCore already initialized")
            return
        }
        
        logger.info("Starting VingiCore initialization...")
        systemStatus = .initializing
        
        do {
            // Store configuration
            self.configuration = config
            
            // Initialize security manager first
            self.securityManager = try await SecurityManager(config: config.securityConfig)
            logger.info("Security manager initialized")
            
            // Initialize context manager
            self.contextManager = try await ContextManager(
                config: config,
                securityManager: securityManager!
            )
            logger.info("Context manager initialized")
            
            // Initialize intelligence engine
            self.intelligenceEngine = try await IntelligenceEngine(
                config: config,
                contextManager: contextManager!
            )
            logger.info("Intelligence engine initialized")
            
            // Initialize automation engine
            self.automationEngine = try await AutomationEngine(
                config: config,
                contextManager: contextManager!,
                intelligenceEngine: intelligenceEngine!
            )
            logger.info("Automation engine initialized")
            
            // Setup inter-component communication
            setupComponentCommunication()
            
            // Mark as initialized
            isInitialized = true
            systemStatus = .ready
            
            logger.info("VingiCore initialization completed successfully")
            
        } catch {
            systemStatus = .error
            logger.error("VingiCore initialization failed: \(error)")
            throw VingiError.initializationFailed(error)
        }
    }
    
    /// Start the Vingi framework
    public func start() async throws {
        guard isInitialized else {
            throw VingiError.notInitialized
        }
        
        guard !isRunning else {
            logger.warning("VingiCore already running")
            return
        }
        
        logger.info("Starting VingiCore...")
        systemStatus = .starting
        
        do {
            // Start core components
            try await contextManager?.start()
            try await intelligenceEngine?.start()
            try await automationEngine?.start()
            
            isRunning = true
            systemStatus = .running
            
            logger.info("VingiCore started successfully")
            
            // Begin initial context analysis
            await performInitialContextAnalysis()
            
        } catch {
            systemStatus = .error
            logger.error("Failed to start VingiCore: \(error)")
            throw VingiError.startupFailed(error)
        }
    }
    
    /// Stop the Vingi framework
    public func stop() async {
        guard isRunning else {
            logger.warning("VingiCore not running")
            return
        }
        
        logger.info("Stopping VingiCore...")
        systemStatus = .stopping
        
        // Stop components in reverse order
        await automationEngine?.stop()
        await intelligenceEngine?.stop()
        await contextManager?.stop()
        
        isRunning = false
        systemStatus = .ready
        
        logger.info("VingiCore stopped")
    }
    
    /// Shutdown the Vingi framework completely
    public func shutdown() async {
        logger.info("Shutting down VingiCore...")
        
        await stop()
        
        // Clean up resources
        cancellables.removeAll()
        contextManager = nil
        intelligenceEngine = nil
        automationEngine = nil
        securityManager = nil
        configuration = nil
        
        isInitialized = false
        systemStatus = .idle
        
        logger.info("VingiCore shutdown complete")
    }
    
    // MARK: - Public API
    
    /// Process a user query or command
    /// - Parameters:
    ///   - query: The user query or command
    ///   - context: Optional additional context
    /// - Returns: Response from the intelligence engine
    public func processQuery(_ query: String, context: [String: Any]? = nil) async throws -> IntelligenceResponse {
        guard isRunning else {
            throw VingiError.notRunning
        }
        
        guard let intelligenceEngine = intelligenceEngine else {
            throw VingiError.componentNotAvailable("IntelligenceEngine")
        }
        
        logger.info("Processing query: \(query)")
        
        let userContext = UserContext(
            query: query,
            timestamp: Date(),
            additionalContext: context ?? [:],
            currentContext: currentContext
        )
        
        let response = try await intelligenceEngine.processQuery(query, context: userContext)
        
        // Update current context
        await updateCurrentContext(from: response)
        
        return response
    }
    
    /// Create an automation rule
    /// - Parameter rule: The automation rule to create
    public func createAutomationRule(_ rule: AutomationRule) async throws {
        guard let automationEngine = automationEngine else {
            throw VingiError.componentNotAvailable("AutomationEngine")
        }
        
        try await automationEngine.createRule(rule)
        logger.info("Created automation rule: \(rule.name)")
    }
    
    /// Get current system insights
    /// - Returns: Array of current insights
    public func getInsights() async throws -> [Insight] {
        guard let intelligenceEngine = intelligenceEngine else {
            throw VingiError.componentNotAvailable("IntelligenceEngine")
        }
        
        return try await intelligenceEngine.generateInsights()
    }
    
    /// Get system health status
    /// - Returns: Current system health
    public func getSystemHealth() async -> SystemHealth {
        var health = SystemHealth()
        
        health.coreStatus = systemStatus
        health.isInitialized = isInitialized
        health.isRunning = isRunning
        
        if let contextManager = contextManager {
            health.contextManagerHealth = await contextManager.getHealth()
        }
        
        if let intelligenceEngine = intelligenceEngine {
            health.intelligenceEngineHealth = await intelligenceEngine.getHealth()
        }
        
        if let automationEngine = automationEngine {
            health.automationEngineHealth = await automationEngine.getHealth()
        }
        
        return health
    }
    
    // MARK: - Private Methods
    
    private func setupComponentCommunication() {
        // Setup communication between components
        contextManager?.contextUpdates
            .sink { [weak self] context in
                Task { @MainActor in
                    self?.currentContext = context
                }
            }
            .store(in: &cancellables)
        
        intelligenceEngine?.insights
            .sink { [weak self] insights in
                Task {
                    await self?.processNewInsights(insights)
                }
            }
            .store(in: &cancellables)
    }
    
    private func performInitialContextAnalysis() async {
        guard let contextManager = contextManager else { return }
        
        do {
            logger.info("Performing initial context analysis...")
            let analysis = try await contextManager.analyzePatterns()
            logger.info("Initial context analysis completed: \(analysis.summary)")
        } catch {
            logger.error("Initial context analysis failed: \(error)")
        }
    }
    
    private func updateCurrentContext(from response: IntelligenceResponse) async {
        // Update current context based on intelligence response
        if let contextManager = contextManager {
            let contextEvent = ContextEvent(
                type: .intelligenceResponse,
                content: response.content,
                metadata: response.metadata,
                timestamp: Date()
            )
            
            do {
                try await contextManager.updateContext(contextEvent)
            } catch {
                logger.error("Failed to update context: \(error)")
            }
        }
    }
    
    private func processNewInsights(_ insights: [Insight]) async {
        // Process new insights from intelligence engine
        logger.info("Processing \(insights.count) new insights")
        
        for insight in insights {
            if insight.actionable, let automationEngine = automationEngine {
                // Create automation suggestions based on insights
                await automationEngine.processInsight(insight)
            }
        }
    }
}

// MARK: - Supporting Types

public enum SystemStatus {
    case idle
    case initializing
    case ready
    case starting
    case running
    case stopping
    case error
}

public struct SystemHealth {
    public var coreStatus: SystemStatus = .idle
    public var isInitialized: Bool = false
    public var isRunning: Bool = false
    public var contextManagerHealth: ComponentHealth?
    public var intelligenceEngineHealth: ComponentHealth?
    public var automationEngineHealth: ComponentHealth?
    
    public var overallHealth: HealthStatus {
        let components = [
            contextManagerHealth?.status,
            intelligenceEngineHealth?.status,
            automationEngineHealth?.status
        ].compactMap { $0 }
        
        if components.contains(.critical) {
            return .critical
        } else if components.contains(.warning) {
            return .warning
        } else if components.allSatisfy({ $0 == .healthy }) {
            return .healthy
        } else {
            return .warning
        }
    }
}

public struct ComponentHealth {
    public let componentName: String
    public let status: HealthStatus
    public let lastUpdate: Date
    public let metrics: [String: Any]
    public let issues: [String]
    
    public init(componentName: String, status: HealthStatus, metrics: [String: Any] = [:], issues: [String] = []) {
        self.componentName = componentName
        self.status = status
        self.lastUpdate = Date()
        self.metrics = metrics
        self.issues = issues
    }
}

public enum HealthStatus {
    case healthy
    case warning
    case critical
}

// MARK: - Error Types

public enum VingiError: Error, LocalizedError {
    case notInitialized
    case notRunning
    case initializationFailed(Error)
    case startupFailed(Error)
    case componentNotAvailable(String)
    case configurationInvalid(String)
    case securityError(String)
    
    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "VingiCore is not initialized. Call initialize(config:) first."
        case .notRunning:
            return "VingiCore is not running. Call start() first."
        case .initializationFailed(let error):
            return "Initialization failed: \(error.localizedDescription)"
        case .startupFailed(let error):
            return "Startup failed: \(error.localizedDescription)"
        case .componentNotAvailable(let component):
            return "Component not available: \(component)"
        case .configurationInvalid(let message):
            return "Invalid configuration: \(message)"
        case .securityError(let message):
            return "Security error: \(message)"
        }
    }
}
