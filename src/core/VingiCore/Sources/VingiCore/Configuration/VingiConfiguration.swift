import Foundation

/// Main configuration structure for Vingi
public struct VingiConfiguration: Codable {
    public let userProfile: UserProfile
    public let privacySettings: PrivacySettings
    public let automationPreferences: AutomationPreferences
    public let securityConfig: SecurityConfiguration
    public let intelligenceConfig: IntelligenceConfiguration
    public let contextConfig: ContextConfiguration
    
    public init(
        userProfile: UserProfile,
        privacySettings: PrivacySettings = PrivacySettings(),
        automationPreferences: AutomationPreferences = AutomationPreferences(),
        securityConfig: SecurityConfiguration = SecurityConfiguration(),
        intelligenceConfig: IntelligenceConfiguration = IntelligenceConfiguration(),
        contextConfig: ContextConfiguration = ContextConfiguration()
    ) {
        self.userProfile = userProfile
        self.privacySettings = privacySettings
        self.automationPreferences = automationPreferences
        self.securityConfig = securityConfig
        self.intelligenceConfig = intelligenceConfig
        self.contextConfig = contextConfig
    }
    
    /// Validate the configuration
    /// - Throws: VingiError.configurationInvalid if validation fails
    public func validate() throws {
        try userProfile.validate()
        try privacySettings.validate()
        try automationPreferences.validate()
        try securityConfig.validate()
        try intelligenceConfig.validate()
        try contextConfig.validate()
    }
}

// MARK: - User Profile

public struct UserProfile: Codable {
    public let name: String
    public let email: String?
    public let expertiseDomains: [String]
    public let workingHours: WorkingHours?
    public let preferences: [String: AnyCodable]
    public let privacyLevel: PrivacyLevel
    
    public init(
        name: String,
        email: String? = nil,
        expertiseDomains: [String] = [],
        workingHours: WorkingHours? = nil,
        preferences: [String: AnyCodable] = [:],
        privacyLevel: PrivacyLevel = .maximum
    ) {
        self.name = name
        self.email = email
        self.expertiseDomains = expertiseDomains
        self.workingHours = workingHours
        self.preferences = preferences
        self.privacyLevel = privacyLevel
    }
    
    func validate() throws {
        guard !name.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw VingiError.configurationInvalid("User name cannot be empty")
        }
        
        if let email = email {
            guard email.contains("@") && email.contains(".") else {
                throw VingiError.configurationInvalid("Invalid email format")
            }
        }
    }
}

public struct WorkingHours: Codable {
    public let startTime: TimeOfDay
    public let endTime: TimeOfDay
    public let timeZone: TimeZone
    public let workDays: Set<Weekday>
    
    public init(
        startTime: TimeOfDay,
        endTime: TimeOfDay,
        timeZone: TimeZone = .current,
        workDays: Set<Weekday> = [.monday, .tuesday, .wednesday, .thursday, .friday]
    ) {
        self.startTime = startTime
        self.endTime = endTime
        self.timeZone = timeZone
        self.workDays = workDays
    }
}

public struct TimeOfDay: Codable {
    public let hour: Int
    public let minute: Int
    
    public init(hour: Int, minute: Int) {
        self.hour = hour
        self.minute = minute
    }
}

public enum Weekday: String, CaseIterable, Codable {
    case monday, tuesday, wednesday, thursday, friday, saturday, sunday
}

// MARK: - Privacy Settings

public struct PrivacySettings: Codable {
    public let privacyLevel: PrivacyLevel
    public let localProcessingOnly: Bool
    public let dataRetentionDays: Int
    public let analyticsEnabled: Bool
    public let crashReportingEnabled: Bool
    public let encryptionRequired: Bool
    public let allowedDataSources: Set<DataSource>
    
    public init(
        privacyLevel: PrivacyLevel = .maximum,
        localProcessingOnly: Bool = true,
        dataRetentionDays: Int = 365,
        analyticsEnabled: Bool = false,
        crashReportingEnabled: Bool = false,
        encryptionRequired: Bool = true,
        allowedDataSources: Set<DataSource> = [.calendar, .files, .email]
    ) {
        self.privacyLevel = privacyLevel
        self.localProcessingOnly = localProcessingOnly
        self.dataRetentionDays = dataRetentionDays
        self.analyticsEnabled = analyticsEnabled
        self.crashReportingEnabled = crashReportingEnabled
        self.encryptionRequired = encryptionRequired
        self.allowedDataSources = allowedDataSources
    }
    
    func validate() throws {
        guard dataRetentionDays > 0 else {
            throw VingiError.configurationInvalid("Data retention days must be positive")
        }
        
        guard dataRetentionDays <= 3650 else { // 10 years max
            throw VingiError.configurationInvalid("Data retention days cannot exceed 10 years")
        }
    }
}

public enum PrivacyLevel: String, CaseIterable, Codable {
    case minimal = "minimal"
    case balanced = "balanced"
    case maximum = "maximum"
    
    public var description: String {
        switch self {
        case .minimal:
            return "Minimal Privacy - Some data may be processed externally for enhanced features"
        case .balanced:
            return "Balanced - Mix of local and external processing with user control"
        case .maximum:
            return "Maximum Privacy - All processing occurs locally, no external data sharing"
        }
    }
}

public enum DataSource: String, CaseIterable, Codable {
    case calendar = "calendar"
    case contacts = "contacts"
    case email = "email"
    case files = "files"
    case messages = "messages"
    case location = "location"
    case usage = "usage"
    case web = "web"
}

// MARK: - Automation Preferences

public struct AutomationPreferences: Codable {
    public let emailManagement: EmailAutomationConfig
    public let fileOrganization: FileAutomationConfig
    public let calendarOptimization: CalendarAutomationConfig
    public let researchAutomation: ResearchAutomationConfig
    public let notificationManagement: NotificationConfig
    
    public init(
        emailManagement: EmailAutomationConfig = EmailAutomationConfig(),
        fileOrganization: FileAutomationConfig = FileAutomationConfig(),
        calendarOptimization: CalendarAutomationConfig = CalendarAutomationConfig(),
        researchAutomation: ResearchAutomationConfig = ResearchAutomationConfig(),
        notificationManagement: NotificationConfig = NotificationConfig()
    ) {
        self.emailManagement = emailManagement
        self.fileOrganization = fileOrganization
        self.calendarOptimization = calendarOptimization
        self.researchAutomation = researchAutomation
        self.notificationManagement = notificationManagement
    }
    
    func validate() throws {
        try emailManagement.validate()
        try fileOrganization.validate()
        try calendarOptimization.validate()
        try researchAutomation.validate()
        try notificationManagement.validate()
    }
}

public struct EmailAutomationConfig: Codable {
    public let enabled: Bool
    public let autoResponse: Bool
    public let smartSorting: Bool
    public let urgencyDetection: Bool
    public let autoArchive: Bool
    public let responseTemplates: [String: String]
    
    public init(
        enabled: Bool = true,
        autoResponse: Bool = false,
        smartSorting: Bool = true,
        urgencyDetection: Bool = true,
        autoArchive: Bool = true,
        responseTemplates: [String: String] = [:]
    ) {
        self.enabled = enabled
        self.autoResponse = autoResponse
        self.smartSorting = smartSorting
        self.urgencyDetection = urgencyDetection
        self.autoArchive = autoArchive
        self.responseTemplates = responseTemplates
    }
    
    func validate() throws {
        // Email automation validation logic
    }
}

public struct FileAutomationConfig: Codable {
    public let enabled: Bool
    public let autoOrganizeDownloads: Bool
    public let semanticNaming: Bool
    public let duplicateDetection: Bool
    public let smartFolders: Bool
    public let watchedDirectories: [String]
    
    public init(
        enabled: Bool = true,
        autoOrganizeDownloads: Bool = true,
        semanticNaming: Bool = true,
        duplicateDetection: Bool = true,
        smartFolders: Bool = true,
        watchedDirectories: [String] = []
    ) {
        self.enabled = enabled
        self.autoOrganizeDownloads = autoOrganizeDownloads
        self.semanticNaming = semanticNaming
        self.duplicateDetection = duplicateDetection
        self.smartFolders = smartFolders
        self.watchedDirectories = watchedDirectories
    }
    
    func validate() throws {
        // File automation validation logic
    }
}

public struct CalendarAutomationConfig: Codable {
    public let enabled: Bool
    public let meetingPreparation: Bool
    public let smartScheduling: Bool
    public let conflictDetection: Bool
    public let travelTimeCalculation: Bool
    public let focusTimeBlocking: Bool
    
    public init(
        enabled: Bool = true,
        meetingPreparation: Bool = true,
        smartScheduling: Bool = true,
        conflictDetection: Bool = true,
        travelTimeCalculation: Bool = false,
        focusTimeBlocking: Bool = true
    ) {
        self.enabled = enabled
        self.meetingPreparation = meetingPreparation
        self.smartScheduling = smartScheduling
        self.conflictDetection = conflictDetection
        self.travelTimeCalculation = travelTimeCalculation
        self.focusTimeBlocking = focusTimeBlocking
    }
    
    func validate() throws {
        // Calendar automation validation logic
    }
}

public struct ResearchAutomationConfig: Codable {
    public let enabled: Bool
    public let sources: Set<ResearchSource>
    public let filters: ResearchFilters
    public let autoSummarization: Bool
    public let relevanceThreshold: Double
    
    public init(
        enabled: Bool = true,
        sources: Set<ResearchSource> = [.arxiv, .scholar, .github],
        filters: ResearchFilters = ResearchFilters(),
        autoSummarization: Bool = true,
        relevanceThreshold: Double = 0.7
    ) {
        self.enabled = enabled
        self.sources = sources
        self.filters = filters
        self.autoSummarization = autoSummarization
        self.relevanceThreshold = relevanceThreshold
    }
    
    func validate() throws {
        guard relevanceThreshold >= 0.0 && relevanceThreshold <= 1.0 else {
            throw VingiError.configurationInvalid("Relevance threshold must be between 0.0 and 1.0")
        }
    }
}

public enum ResearchSource: String, CaseIterable, Codable {
    case arxiv = "arxiv"
    case scholar = "google_scholar"
    case github = "github"
    case pubmed = "pubmed"
    case semanticScholar = "semantic_scholar"
    case techBlogs = "tech_blogs"
}

public struct ResearchFilters: Codable {
    public let minRelevanceScore: Double
    public let maxResultsPerQuery: Int
    public let includePreprints: Bool
    public let dateRange: DateRange?
    public let excludedSources: Set<String>
    
    public init(
        minRelevanceScore: Double = 0.6,
        maxResultsPerQuery: Int = 10,
        includePreprints: Bool = true,
        dateRange: DateRange? = nil,
        excludedSources: Set<String> = []
    ) {
        self.minRelevanceScore = minRelevanceScore
        self.maxResultsPerQuery = maxResultsPerQuery
        self.includePreprints = includePreprints
        self.dateRange = dateRange
        self.excludedSources = excludedSources
    }
}

public struct DateRange: Codable {
    public let start: Date?
    public let end: Date?
    
    public init(start: Date? = nil, end: Date? = nil) {
        self.start = start
        self.end = end
    }
}

public struct NotificationConfig: Codable {
    public let enabled: Bool
    public let urgentOnly: Bool
    public let quietHours: QuietHours?
    public let channels: Set<NotificationChannel>
    
    public init(
        enabled: Bool = true,
        urgentOnly: Bool = false,
        quietHours: QuietHours? = nil,
        channels: Set<NotificationChannel> = [.system, .menuBar]
    ) {
        self.enabled = enabled
        self.urgentOnly = urgentOnly
        self.quietHours = quietHours
        self.channels = channels
    }
    
    func validate() throws {
        // Notification validation logic
    }
}

public struct QuietHours: Codable {
    public let startTime: TimeOfDay
    public let endTime: TimeOfDay
    public let days: Set<Weekday>
    
    public init(startTime: TimeOfDay, endTime: TimeOfDay, days: Set<Weekday>) {
        self.startTime = startTime
        self.endTime = endTime
        self.days = days
    }
}

public enum NotificationChannel: String, CaseIterable, Codable {
    case system = "system"
    case menuBar = "menu_bar"
    case email = "email"
    case sound = "sound"
}

// MARK: - Security Configuration

public struct SecurityConfiguration: Codable {
    public let encryptionKeyRotationDays: Int
    public let requireBiometrics: Bool
    public let dataRetentionDays: Int
    public let auditLogging: Bool
    public let secureConnectionsOnly: Bool
    public let encryptionAlgorithm: EncryptionAlgorithm
    
    public init(
        encryptionKeyRotationDays: Int = 30,
        requireBiometrics: Bool = true,
        dataRetentionDays: Int = 365,
        auditLogging: Bool = true,
        secureConnectionsOnly: Bool = true,
        encryptionAlgorithm: EncryptionAlgorithm = .chaCha20Poly1305
    ) {
        self.encryptionKeyRotationDays = encryptionKeyRotationDays
        self.requireBiometrics = requireBiometrics
        self.dataRetentionDays = dataRetentionDays
        self.auditLogging = auditLogging
        self.secureConnectionsOnly = secureConnectionsOnly
        self.encryptionAlgorithm = encryptionAlgorithm
    }
    
    func validate() throws {
        guard encryptionKeyRotationDays > 0 else {
            throw VingiError.configurationInvalid("Encryption key rotation days must be positive")
        }
        
        guard dataRetentionDays > 0 else {
            throw VingiError.configurationInvalid("Data retention days must be positive")
        }
    }
}

public enum EncryptionAlgorithm: String, CaseIterable, Codable {
    case chaCha20Poly1305 = "chacha20_poly1305"
    case aes256GCM = "aes256_gcm"
}

// MARK: - Intelligence Configuration

public struct IntelligenceConfiguration: Codable {
    public let modelConfigs: [String: ModelConfig]
    public let processingLimits: ProcessingLimits
    public let cachingStrategy: CachingStrategy
    public let confidenceThresholds: ConfidenceThresholds
    
    public init(
        modelConfigs: [String: ModelConfig] = [:],
        processingLimits: ProcessingLimits = ProcessingLimits(),
        cachingStrategy: CachingStrategy = CachingStrategy(),
        confidenceThresholds: ConfidenceThresholds = ConfidenceThresholds()
    ) {
        self.modelConfigs = modelConfigs
        self.processingLimits = processingLimits
        self.cachingStrategy = cachingStrategy
        self.confidenceThresholds = confidenceThresholds
    }
    
    func validate() throws {
        try processingLimits.validate()
        try cachingStrategy.validate()
        try confidenceThresholds.validate()
    }
}

public struct ModelConfig: Codable {
    public let modelPath: String
    public let modelType: ModelType
    public let parameters: [String: AnyCodable]
    public let enabled: Bool
    
    public init(modelPath: String, modelType: ModelType, parameters: [String: AnyCodable] = [:], enabled: Bool = true) {
        self.modelPath = modelPath
        self.modelType = modelType
        self.parameters = parameters
        self.enabled = enabled
    }
}

public enum ModelType: String, CaseIterable, Codable {
    case embedding = "embedding"
    case classification = "classification"
    case generation = "generation"
    case summarization = "summarization"
}

public struct ProcessingLimits: Codable {
    public let maxConcurrentTasks: Int
    public let maxMemoryUsageMB: Int
    public let maxProcessingTimeSeconds: Int
    public let batchSize: Int
    
    public init(
        maxConcurrentTasks: Int = 10,
        maxMemoryUsageMB: Int = 1024,
        maxProcessingTimeSeconds: Int = 30,
        batchSize: Int = 32
    ) {
        self.maxConcurrentTasks = maxConcurrentTasks
        self.maxMemoryUsageMB = maxMemoryUsageMB
        self.maxProcessingTimeSeconds = maxProcessingTimeSeconds
        self.batchSize = batchSize
    }
    
    func validate() throws {
        guard maxConcurrentTasks > 0 else {
            throw VingiError.configurationInvalid("Max concurrent tasks must be positive")
        }
        
        guard maxMemoryUsageMB > 0 else {
            throw VingiError.configurationInvalid("Max memory usage must be positive")
        }
    }
}

public struct CachingStrategy: Codable {
    public let enabled: Bool
    public let maxCacheSizeMB: Int
    public let cacheExpiryHours: Int
    public let persistToDisk: Bool
    
    public init(
        enabled: Bool = true,
        maxCacheSizeMB: Int = 512,
        cacheExpiryHours: Int = 24,
        persistToDisk: Bool = true
    ) {
        self.enabled = enabled
        self.maxCacheSizeMB = maxCacheSizeMB
        self.cacheExpiryHours = cacheExpiryHours
        self.persistToDisk = persistToDisk
    }
    
    func validate() throws {
        guard maxCacheSizeMB > 0 else {
            throw VingiError.configurationInvalid("Cache size must be positive")
        }
    }
}

public struct ConfidenceThresholds: Codable {
    public let classification: Double
    public let prediction: Double
    public let automation: Double
    
    public init(
        classification: Double = 0.8,
        prediction: Double = 0.7,
        automation: Double = 0.9
    ) {
        self.classification = classification
        self.prediction = prediction
        self.automation = automation
    }
    
    func validate() throws {
        let thresholds = [classification, prediction, automation]
        for threshold in thresholds {
            guard threshold >= 0.0 && threshold <= 1.0 else {
                throw VingiError.configurationInvalid("Confidence thresholds must be between 0.0 and 1.0")
            }
        }
    }
}

// MARK: - Context Configuration

public struct ContextConfiguration: Codable {
    public let maxContextNodes: Int
    public let contextRetentionDays: Int
    public let relationshipStrengthThreshold: Double
    public let patternDetectionConfig: PatternDetectionConfig
    
    public init(
        maxContextNodes: Int = 10000,
        contextRetentionDays: Int = 365,
        relationshipStrengthThreshold: Double = 0.5,
        patternDetectionConfig: PatternDetectionConfig = PatternDetectionConfig()
    ) {
        self.maxContextNodes = maxContextNodes
        self.contextRetentionDays = contextRetentionDays
        self.relationshipStrengthThreshold = relationshipStrengthThreshold
        self.patternDetectionConfig = patternDetectionConfig
    }
    
    func validate() throws {
        guard maxContextNodes > 0 else {
            throw VingiError.configurationInvalid("Max context nodes must be positive")
        }
        
        guard contextRetentionDays > 0 else {
            throw VingiError.configurationInvalid("Context retention days must be positive")
        }
        
        guard relationshipStrengthThreshold >= 0.0 && relationshipStrengthThreshold <= 1.0 else {
            throw VingiError.configurationInvalid("Relationship strength threshold must be between 0.0 and 1.0")
        }
    }
}

public struct PatternDetectionConfig: Codable {
    public let enabled: Bool
    public let minPatternOccurrences: Int
    public let analysisWindowDays: Int
    public let patternTypes: Set<PatternType>
    
    public init(
        enabled: Bool = true,
        minPatternOccurrences: Int = 3,
        analysisWindowDays: Int = 30,
        patternTypes: Set<PatternType> = [.temporal, .behavioral, .contextual]
    ) {
        self.enabled = enabled
        self.minPatternOccurrences = minPatternOccurrences
        self.analysisWindowDays = analysisWindowDays
        self.patternTypes = patternTypes
    }
}

public enum PatternType: String, CaseIterable, Codable {
    case temporal = "temporal"
    case behavioral = "behavioral"
    case contextual = "contextual"
    case social = "social"
}

// MARK: - Supporting Types

/// Type-erased codable wrapper for configuration values
public struct AnyCodable: Codable {
    public let value: Any
    
    public init<T: Codable>(_ value: T) {
        self.value = value
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        
        if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dictionary = try? container.decode([String: AnyCodable].self) {
            value = dictionary.mapValues { $0.value }
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported type for AnyCodable"
            )
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        
        switch value {
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [Any]:
            let anyCodableArray = array.map { AnyCodable($0 as! Codable) }
            try container.encode(anyCodableArray)
        case let dictionary as [String: Any]:
            let anyCodableDict = dictionary.mapValues { AnyCodable($0 as! Codable) }
            try container.encode(anyCodableDict)
        default:
            throw EncodingError.invalidValue(
                value,
                EncodingError.Context(codingPath: [], debugDescription: "Unsupported type for AnyCodable")
            )
        }
    }
} 