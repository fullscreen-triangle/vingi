import Foundation
import Combine

// MARK: - User Context

public struct UserContext: Codable, Identifiable {
    public let id: UUID
    public let query: String?
    public let timestamp: Date
    public let additionalContext: [String: Any]
    public let currentContext: UserContext?
    public let sessionId: UUID
    public let deviceInfo: DeviceInfo?
    
    public init(
        id: UUID = UUID(),
        query: String? = nil,
        timestamp: Date = Date(),
        additionalContext: [String: Any] = [:],
        currentContext: UserContext? = nil,
        sessionId: UUID = UUID(),
        deviceInfo: DeviceInfo? = nil
    ) {
        self.id = id
        self.query = query
        self.timestamp = timestamp
        self.additionalContext = additionalContext
        self.currentContext = currentContext
        self.sessionId = sessionId
        self.deviceInfo = deviceInfo
    }
    
    // Custom Codable implementation due to [String: Any]
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        query = try container.decodeIfPresent(String.self, forKey: .query)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        sessionId = try container.decode(UUID.self, forKey: .sessionId)
        deviceInfo = try container.decodeIfPresent(DeviceInfo.self, forKey: .deviceInfo)
        currentContext = try container.decodeIfPresent(UserContext.self, forKey: .currentContext)
        
        // Decode additionalContext as Data and convert
        if let contextData = try container.decodeIfPresent(Data.self, forKey: .additionalContext) {
            additionalContext = (try? JSONSerialization.jsonObject(with: contextData) as? [String: Any]) ?? [:]
        } else {
            additionalContext = [:]
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encodeIfPresent(query, forKey: .query)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(sessionId, forKey: .sessionId)
        try container.encodeIfPresent(deviceInfo, forKey: .deviceInfo)
        try container.encodeIfPresent(currentContext, forKey: .currentContext)
        
        // Encode additionalContext as Data
        let contextData = try JSONSerialization.data(withJSONObject: additionalContext)
        try container.encode(contextData, forKey: .additionalContext)
    }
    
    private enum CodingKeys: String, CodingKey {
        case id, query, timestamp, additionalContext, currentContext, sessionId, deviceInfo
    }
}

public struct DeviceInfo: Codable {
    public let deviceType: DeviceType
    public let osVersion: String
    public let appVersion: String
    public let deviceId: String
    
    public init(deviceType: DeviceType, osVersion: String, appVersion: String, deviceId: String) {
        self.deviceType = deviceType
        self.osVersion = osVersion
        self.appVersion = appVersion
        self.deviceId = deviceId
    }
}

public enum DeviceType: String, Codable {
    case mac = "mac"
    case iPhone = "iphone"
    case iPad = "ipad"
    case unknown = "unknown"
}

// MARK: - Context Events and Nodes

public struct ContextEvent: Codable, Identifiable {
    public let id: UUID
    public let type: ContextEventType
    public let content: String
    public let metadata: [String: String]
    public let timestamp: Date
    public let sourceId: String?
    public let relevanceScore: Double?
    
    public init(
        id: UUID = UUID(),
        type: ContextEventType,
        content: String,
        metadata: [String: String] = [:],
        timestamp: Date = Date(),
        sourceId: String? = nil,
        relevanceScore: Double? = nil
    ) {
        self.id = id
        self.type = type
        self.content = content
        self.metadata = metadata
        self.timestamp = timestamp
        self.sourceId = sourceId
        self.relevanceScore = relevanceScore
    }
}

public enum ContextEventType: String, CaseIterable, Codable {
    case emailReceived = "email_received"
    case emailSent = "email_sent"
    case fileCreated = "file_created"
    case fileModified = "file_modified"
    case meetingScheduled = "meeting_scheduled"
    case meetingStarted = "meeting_started"
    case meetingEnded = "meeting_ended"
    case applicationLaunched = "app_launched"
    case applicationClosed = "app_closed"
    case queryProcessed = "query_processed"
    case intelligenceResponse = "intelligence_response"
    case automationTriggered = "automation_triggered"
    case userInteraction = "user_interaction"
    case systemEvent = "system_event"
}

public struct ContextNode: Codable, Identifiable {
    public let id: UUID
    public let type: ContextNodeType
    public let content: String
    public let contentHash: String
    public let timestamp: Date
    public let metadata: [String: String]
    public let relationships: [ContextRelationship]
    public let relevanceScore: Double
    public let accessCount: Int
    public let lastAccessed: Date?
    
    public init(
        id: UUID = UUID(),
        type: ContextNodeType,
        content: String,
        timestamp: Date = Date(),
        metadata: [String: String] = [:],
        relationships: [ContextRelationship] = [],
        relevanceScore: Double = 0.0,
        accessCount: Int = 0,
        lastAccessed: Date? = nil
    ) {
        self.id = id
        self.type = type
        self.content = content
        self.contentHash = content.sha256
        self.timestamp = timestamp
        self.metadata = metadata
        self.relationships = relationships
        self.relevanceScore = relevanceScore
        self.accessCount = accessCount
        self.lastAccessed = lastAccessed
    }
}

public enum ContextNodeType: String, CaseIterable, Codable {
    case person = "person"
    case email = "email"
    case meeting = "meeting"
    case file = "file"
    case task = "task"
    case project = "project"
    case research = "research"
    case insight = "insight"
    case automation = "automation"
}

public struct ContextRelationship: Codable, Identifiable {
    public let id: UUID
    public let sourceId: UUID
    public let targetId: UUID
    public let type: RelationshipType
    public let strength: Double
    public let metadata: [String: String]
    public let createdAt: Date
    public let lastUpdated: Date
    
    public init(
        id: UUID = UUID(),
        sourceId: UUID,
        targetId: UUID,
        type: RelationshipType,
        strength: Double,
        metadata: [String: String] = [:],
        createdAt: Date = Date(),
        lastUpdated: Date = Date()
    ) {
        self.id = id
        self.sourceId = sourceId
        self.targetId = targetId
        self.type = type
        self.strength = strength
        self.metadata = metadata
        self.createdAt = createdAt
        self.lastUpdated = lastUpdated
    }
}

public enum RelationshipType: String, CaseIterable, Codable {
    case contains = "contains"
    case relatedTo = "related_to"
    case dependsOn = "depends_on"
    case causedBy = "caused_by"
    case similarTo = "similar_to"
    case mentions = "mentions"
    case attendedBy = "attended_by"
    case createdBy = "created_by"
    case triggeredBy = "triggered_by"
}

// MARK: - Intelligence Types

public struct IntelligenceResponse: Codable, Identifiable {
    public let id: UUID
    public let query: String
    public let content: String
    public let confidence: Double
    public let sources: [String]
    public let metadata: [String: String]
    public let timestamp: Date
    public let processingTime: TimeInterval
    public let suggestions: [ActionSuggestion]
    
    public init(
        id: UUID = UUID(),
        query: String,
        content: String,
        confidence: Double,
        sources: [String] = [],
        metadata: [String: String] = [:],
        timestamp: Date = Date(),
        processingTime: TimeInterval = 0,
        suggestions: [ActionSuggestion] = []
    ) {
        self.id = id
        self.query = query
        self.content = content
        self.confidence = confidence
        self.sources = sources
        self.metadata = metadata
        self.timestamp = timestamp
        self.processingTime = processingTime
        self.suggestions = suggestions
    }
}

public struct ActionSuggestion: Codable, Identifiable {
    public let id: UUID
    public let title: String
    public let description: String
    public let actionType: ActionType
    public let confidence: Double
    public let parameters: [String: String]
    public let estimatedBenefit: Double
    
    public init(
        id: UUID = UUID(),
        title: String,
        description: String,
        actionType: ActionType,
        confidence: Double,
        parameters: [String: String] = [:],
        estimatedBenefit: Double = 0.0
    ) {
        self.id = id
        self.title = title
        self.description = description
        self.actionType = actionType
        self.confidence = confidence
        self.parameters = parameters
        self.estimatedBenefit = estimatedBenefit
    }
}

public enum ActionType: String, CaseIterable, Codable {
    case createAutomation = "create_automation"
    case organizeFiles = "organize_files"
    case scheduleTask = "schedule_task"
    case sendEmail = "send_email"
    case createMeeting = "create_meeting"
    case researchTopic = "research_topic"
    case summarizeContent = "summarize_content"
    case setReminder = "set_reminder"
}

public struct Insight: Codable, Identifiable {
    public let id: UUID
    public let title: String
    public let description: String
    public let category: InsightCategory
    public let confidence: Double
    public let actionable: Bool
    public let metadata: [String: String]
    public let timestamp: Date
    public let relatedContextIds: [UUID]
    public let suggestedActions: [ActionSuggestion]
    
    public init(
        id: UUID = UUID(),
        title: String,
        description: String,
        category: InsightCategory,
        confidence: Double,
        actionable: Bool = false,
        metadata: [String: String] = [:],
        timestamp: Date = Date(),
        relatedContextIds: [UUID] = [],
        suggestedActions: [ActionSuggestion] = []
    ) {
        self.id = id
        self.title = title
        self.description = description
        self.category = category
        self.confidence = confidence
        self.actionable = actionable
        self.metadata = metadata
        self.timestamp = timestamp
        self.relatedContextIds = relatedContextIds
        self.suggestedActions = suggestedActions
    }
}

public enum InsightCategory: String, CaseIterable, Codable {
    case productivity = "productivity"
    case timeManagement = "time_management"
    case communication = "communication"
    case learning = "learning"
    case wellbeing = "wellbeing"
    case efficiency = "efficiency"
    case collaboration = "collaboration"
    case research = "research"
}

// MARK: - Automation Types

public struct AutomationRule: Codable, Identifiable {
    public let id: UUID
    public let name: String
    public let description: String
    public let trigger: AutomationTrigger
    public let conditions: [AutomationCondition]
    public let actions: [AutomationAction]
    public let enabled: Bool
    public let createdAt: Date
    public let lastTriggered: Date?
    public let triggerCount: Int
    public let successRate: Double
    
    public init(
        id: UUID = UUID(),
        name: String,
        description: String,
        trigger: AutomationTrigger,
        conditions: [AutomationCondition] = [],
        actions: [AutomationAction],
        enabled: Bool = true,
        createdAt: Date = Date(),
        lastTriggered: Date? = nil,
        triggerCount: Int = 0,
        successRate: Double = 0.0
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.trigger = trigger
        self.conditions = conditions
        self.actions = actions
        self.enabled = enabled
        self.createdAt = createdAt
        self.lastTriggered = lastTriggered
        self.triggerCount = triggerCount
        self.successRate = successRate
    }
}

public struct AutomationTrigger: Codable {
    public let type: TriggerType
    public let parameters: [String: String]
    public let schedule: Schedule?
    
    public init(type: TriggerType, parameters: [String: String] = [:], schedule: Schedule? = nil) {
        self.type = type
        self.parameters = parameters
        self.schedule = schedule
    }
}

public enum TriggerType: String, CaseIterable, Codable {
    case emailReceived = "email_received"
    case fileCreated = "file_created"
    case calendarEvent = "calendar_event"
    case timeSchedule = "time_schedule"
    case contextPattern = "context_pattern"
    case userQuery = "user_query"
    case systemEvent = "system_event"
}

public struct Schedule: Codable {
    public let type: ScheduleType
    public let interval: TimeInterval?
    public let cronExpression: String?
    public let daysOfWeek: Set<Weekday>?
    public let timeOfDay: TimeOfDay?
    
    public init(
        type: ScheduleType,
        interval: TimeInterval? = nil,
        cronExpression: String? = nil,
        daysOfWeek: Set<Weekday>? = nil,
        timeOfDay: TimeOfDay? = nil
    ) {
        self.type = type
        self.interval = interval
        self.cronExpression = cronExpression
        self.daysOfWeek = daysOfWeek
        self.timeOfDay = timeOfDay
    }
}

public enum ScheduleType: String, CaseIterable, Codable {
    case interval = "interval"
    case daily = "daily"
    case weekly = "weekly"
    case monthly = "monthly"
    case cron = "cron"
}

public struct AutomationCondition: Codable, Identifiable {
    public let id: UUID
    public let type: ConditionType
    public let field: String
    public let operator: ConditionOperator
    public let value: String
    public let negate: Bool
    
    public init(
        id: UUID = UUID(),
        type: ConditionType,
        field: String,
        operator: ConditionOperator,
        value: String,
        negate: Bool = false
    ) {
        self.id = id
        self.type = type
        self.field = field
        self.operator = `operator`
        self.value = value
        self.negate = negate
    }
}

public enum ConditionType: String, CaseIterable, Codable {
    case context = "context"
    case time = "time"
    case location = "location"
    case content = "content"
    case metadata = "metadata"
}

public enum ConditionOperator: String, CaseIterable, Codable {
    case equals = "equals"
    case contains = "contains"
    case startsWith = "starts_with"
    case endsWith = "ends_with"
    case greaterThan = "greater_than"
    case lessThan = "less_than"
    case matches = "matches"
}

public struct AutomationAction: Codable, Identifiable {
    public let id: UUID
    public let type: ActionType
    public let parameters: [String: String]
    public let order: Int
    public let enabled: Bool
    
    public init(
        id: UUID = UUID(),
        type: ActionType,
        parameters: [String: String] = [:],
        order: Int = 0,
        enabled: Bool = true
    ) {
        self.id = id
        self.type = type
        self.parameters = parameters
        self.order = order
        self.enabled = enabled
    }
}

public struct AutomationResult: Codable, Identifiable {
    public let id: UUID
    public let ruleId: UUID
    public let success: Bool
    public let message: String
    public let timestamp: Date
    public let executionTime: TimeInterval
    public let actionsExecuted: [UUID]
    public let error: String?
    
    public init(
        id: UUID = UUID(),
        ruleId: UUID,
        success: Bool,
        message: String,
        timestamp: Date = Date(),
        executionTime: TimeInterval,
        actionsExecuted: [UUID] = [],
        error: String? = nil
    ) {
        self.id = id
        self.ruleId = ruleId
        self.success = success
        self.message = message
        self.timestamp = timestamp
        self.executionTime = executionTime
        self.actionsExecuted = actionsExecuted
        self.error = error
    }
}

// MARK: - Pattern Analysis Types

public struct PatternAnalysis: Codable {
    public let patterns: [DetectedPattern]
    public let summary: String
    public let confidence: Double
    public let timestamp: Date
    public let analysisWindow: DateInterval
    
    public init(
        patterns: [DetectedPattern],
        summary: String,
        confidence: Double,
        timestamp: Date = Date(),
        analysisWindow: DateInterval
    ) {
        self.patterns = patterns
        self.summary = summary
        self.confidence = confidence
        self.timestamp = timestamp
        self.analysisWindow = analysisWindow
    }
}

public struct DetectedPattern: Codable, Identifiable {
    public let id: UUID
    public let type: PatternType
    public let description: String
    public let frequency: Double
    public let confidence: Double
    public let significance: Double
    public let examples: [String]
    public let metadata: [String: String]
    
    public init(
        id: UUID = UUID(),
        type: PatternType,
        description: String,
        frequency: Double,
        confidence: Double,
        significance: Double,
        examples: [String] = [],
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.type = type
        self.description = description
        self.frequency = frequency
        self.confidence = confidence
        self.significance = significance
        self.examples = examples
        self.metadata = metadata
    }
}

// MARK: - Research Types

public struct ResearchQuery: Codable, Identifiable {
    public let id: UUID
    public let query: String
    public let sources: Set<ResearchSource>
    public let filters: ResearchFilters
    public let userContext: UserContext?
    public let timestamp: Date
    
    public init(
        id: UUID = UUID(),
        query: String,
        sources: Set<ResearchSource>,
        filters: ResearchFilters,
        userContext: UserContext? = nil,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.query = query
        self.sources = sources
        self.filters = filters
        self.userContext = userContext
        self.timestamp = timestamp
    }
}

public struct ResearchResult: Codable, Identifiable {
    public let id: UUID
    public let query: ResearchQuery
    public let results: [ResearchItem]
    public let summary: String?
    public let totalResults: Int
    public let processingTime: TimeInterval
    public let timestamp: Date
    
    public init(
        id: UUID = UUID(),
        query: ResearchQuery,
        results: [ResearchItem],
        summary: String? = nil,
        totalResults: Int,
        processingTime: TimeInterval,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.query = query
        self.results = results
        self.summary = summary
        self.totalResults = totalResults
        self.processingTime = processingTime
        self.timestamp = timestamp
    }
}

public struct ResearchItem: Codable, Identifiable {
    public let id: UUID
    public let title: String
    public let abstract: String?
    public let authors: [String]
    public let source: ResearchSource
    public let url: URL?
    public let publishedDate: Date?
    public let relevanceScore: Double
    public let citations: Int?
    public let metadata: [String: String]
    
    public init(
        id: UUID = UUID(),
        title: String,
        abstract: String? = nil,
        authors: [String] = [],
        source: ResearchSource,
        url: URL? = nil,
        publishedDate: Date? = nil,
        relevanceScore: Double,
        citations: Int? = nil,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.source = source
        self.url = url
        self.publishedDate = publishedDate
        self.relevanceScore = relevanceScore
        self.citations = citations
        self.metadata = metadata
    }
}

// MARK: - String Extension for Hashing

extension String {
    var sha256: String {
        let data = Data(self.utf8)
        let hashed = data.sha256
        return hashed.map { String(format: "%02hhx", $0) }.joined()
    }
}

extension Data {
    var sha256: Data {
        var hash = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        self.withUnsafeBytes {
            _ = CC_SHA256($0.baseAddress, CC_LONG(self.count), &hash)
        }
        return Data(hash)
    }
}

import CommonCrypto

public enum CognitivePattern {
    case analysisParalysis(
        severity: ParalysisSeverity,
        triggerDomain: String,
        suggestedInterventions: [String]
    )
    case tunnelVision(
        focusDomain: String,
        neglectedDomains: [String],
        riskLevel: TunnelRisk
    )
    case defaultLoop(
        domain: String,
        constraintType: ConstraintType,
        optimizationPotential: Double
    )
    case exceptionalAbilitySelfDoubt(
        ability: SelectiveAbilityRecognitionEngine.AbilityDomain,
        evidenceCount: Int,
        actualPerformance: Double,
        perceivedPerformance: Double,
        confidenceGap: Double
    )
}

public enum PatternType: String, CaseIterable, Codable {
    case temporal = "temporal"
    case behavioral = "behavioral"
    case contextual = "contextual"
    case social = "social"
    case selectiveAbility = "selective_ability"
}

// MARK: - Cognitive Pattern Supporting Types

public enum ParalysisSeverity: String, CaseIterable, Codable {
    case mild = "mild"
    case moderate = "moderate"
    case severe = "severe"
    case critical = "critical"
    
    public var maxComplexity: Int {
        switch self {
        case .mild: return 5
        case .moderate: return 3
        case .severe: return 2
        case .critical: return 1
        }
    }
}

public enum TunnelRisk: String, CaseIterable, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

public enum ConstraintType: String, CaseIterable, Codable {
    case location = "location"
    case time = "time"
    case budget = "budget"
    case convenience = "convenience"
    case familiarity = "familiarity"
    case safety = "safety"
}

public struct UserBehaviorData: Codable {
    public let recentDecisions: [DecisionEvent]
    public let planningActivity: [PlanningEvent]
    public let routineChoices: [ChoiceEvent]
    public let performanceHistory: [SelectiveAbilityRecognitionEngine.PerformanceEvent]
    public let timestamp: Date
    
    public init(
        recentDecisions: [DecisionEvent],
        planningActivity: [PlanningEvent],
        routineChoices: [ChoiceEvent],
        performanceHistory: [SelectiveAbilityRecognitionEngine.PerformanceEvent],
        timestamp: Date = Date()
    ) {
        self.recentDecisions = recentDecisions
        self.planningActivity = planningActivity
        self.routineChoices = routineChoices
        self.performanceHistory = performanceHistory
        self.timestamp = timestamp
    }
}

public struct DecisionEvent: Codable, Identifiable {
    public let id: UUID
    public let domain: String
    public let researchTime: TimeInterval
    public let optionsConsidered: Int
    public let decisionMade: Bool
    public let satisfaction: Double?
    public let timestamp: Date
    
    public init(
        id: UUID = UUID(),
        domain: String,
        researchTime: TimeInterval,
        optionsConsidered: Int,
        decisionMade: Bool,
        satisfaction: Double? = nil,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.domain = domain
        self.researchTime = researchTime
        self.optionsConsidered = optionsConsidered
        self.decisionMade = decisionMade
        self.satisfaction = satisfaction
        self.timestamp = timestamp
    }
}

public struct PlanningEvent: Codable, Identifiable {
    public let id: UUID
    public let activity: String
    public let domain: String
    public let timeSpent: TimeInterval
    public let detailLevel: Double // 0-1
    public let completeness: Double // 0-1
    public let timestamp: Date
    
    public init(
        id: UUID = UUID(),
        activity: String,
        domain: String,
        timeSpent: TimeInterval,
        detailLevel: Double,
        completeness: Double,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.activity = activity
        self.domain = domain
        self.timeSpent = timeSpent
        self.detailLevel = detailLevel
        self.completeness = completeness
        self.timestamp = timestamp
    }
}

public struct CognitiveLoadAssessment: Codable {
    public let overallLoad: Double
    public let patternRisks: [String: Double]
    public let recommendedInterventions: [String]
    public let urgencyLevel: UrgencyLevel
    public let timestamp: Date
    
    public init(
        overallLoad: Double,
        patternRisks: [String: Double],
        recommendedInterventions: [String],
        urgencyLevel: UrgencyLevel,
        timestamp: Date = Date()
    ) {
        self.overallLoad = overallLoad
        self.patternRisks = patternRisks
        self.recommendedInterventions = recommendedInterventions
        self.urgencyLevel = urgencyLevel
        self.timestamp = timestamp
    }
}

public enum UrgencyLevel: String, CaseIterable, Codable {
    case monitoring = "monitoring"
    case attention = "attention"
    case intervention = "intervention"
    case emergency = "emergency"
}

// MARK: - Choice Event Types

public struct ChoiceEvent: Codable, Identifiable {
    public let id: UUID
    public let domain: String
    public let choice: String
    public let alternatives: [String]
    public let satisfaction: Double?
    public let timestamp: Date
    public let metadata: [String: String]
    
    public init(
        id: UUID = UUID(),
        domain: String,
        choice: String,
        alternatives: [String] = [],
        satisfaction: Double? = nil,
        timestamp: Date = Date(),
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.domain = domain
        self.choice = choice
        self.alternatives = alternatives
        self.satisfaction = satisfaction
        self.timestamp = timestamp
        self.metadata = metadata
    }
}

// MARK: - Time-related Types

public enum Weekday: String, CaseIterable, Codable {
    case monday = "monday"
    case tuesday = "tuesday"
    case wednesday = "wednesday"
    case thursday = "thursday"
    case friday = "friday"
    case saturday = "saturday"
    case sunday = "sunday"
}

public struct TimeOfDay: Codable {
    public let hour: Int
    public let minute: Int
    
    public init(hour: Int, minute: Int) {
        self.hour = hour
        self.minute = minute
    }
}

// MARK: - Research Supporting Types

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