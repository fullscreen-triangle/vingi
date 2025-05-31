import Foundation
import Combine

/// Engine that recognizes and validates selective exceptional cognitive abilities
/// Addresses the pattern where users doubt proven strengths due to social expectations
public class SelectiveAbilityRecognitionEngine: ObservableObject {
    
    // MARK: - Types
    
    public struct AbilityMap: Codable {
        public let domains: [AbilityDomain: PerformanceProfile]
        public let overallConfidence: Double
        public let lastUpdated: Date
        
        public init(
            domains: [AbilityDomain: PerformanceProfile],
            overallConfidence: Double,
            lastUpdated: Date = Date()
        ) {
            self.domains = domains
            self.overallConfidence = overallConfidence
            self.lastUpdated = lastUpdated
        }
    }
    
    public struct PerformanceProfile: Codable {
        public let averageAccuracy: Double
        public let contextualFactors: [SituationalFactor]
        public let optimalConditions: [ActivationContext]
        public let confidenceCalibration: Double
        public let evidenceCount: Int
        
        public init(
            averageAccuracy: Double,
            contextualFactors: [SituationalFactor],
            optimalConditions: [ActivationContext],
            confidenceCalibration: Double,
            evidenceCount: Int
        ) {
            self.averageAccuracy = averageAccuracy
            self.contextualFactors = contextualFactors
            self.optimalConditions = optimalConditions
            self.confidenceCalibration = confidenceCalibration
            self.evidenceCount = evidenceCount
        }
    }
    
    public enum AbilityDomain: String, CaseIterable, Codable {
        case meaningfulSequences = "meaningful_sequences"
        case pressureSituations = "pressure_situations"
        case emotionalMemories = "emotional_memories"
        case routineInformation = "routine_information"
        case proceduralMemory = "procedural_memory"
        case spatialNavigation = "spatial_navigation"
        case socialContexts = "social_contexts"
        case temporalSequencing = "temporal_sequencing"
        
        public var description: String {
            switch self {
            case .meaningfulSequences: return "Meaningful Sequences"
            case .pressureSituations: return "High-Pressure Recall"
            case .emotionalMemories: return "Emotional Memories"
            case .routineInformation: return "Routine Information"
            case .proceduralMemory: return "Procedural Memory"
            case .spatialNavigation: return "Spatial Navigation"
            case .socialContexts: return "Social Contexts"
            case .temporalSequencing: return "Temporal Sequencing"
            }
        }
    }
    
    public struct ActivationContext: Codable, Identifiable {
        public let id: UUID
        public let contextType: ContextType
        public let description: String
        public let activationProbability: Double
        public let requiredConditions: [String]
        
        public init(
            id: UUID = UUID(),
            contextType: ContextType,
            description: String,
            activationProbability: Double,
            requiredConditions: [String]
        ) {
            self.id = id
            self.contextType = contextType
            self.description = description
            self.activationProbability = activationProbability
            self.requiredConditions = requiredConditions
        }
    }
    
    public enum ContextType: String, CaseIterable, Codable {
        case highStakes = "high_stakes"
        case timePressed = "time_pressed"
        case emotionallyCharged = "emotionally_charged"
        case socialPressure = "social_pressure"
        case quietReflection = "quiet_reflection"
        case routineTask = "routine_task"
        case novelSituation = "novel_situation"
        
        public var description: String {
            switch self {
            case .highStakes: return "High-Stakes Situations"
            case .timePressed: return "Time-Pressed Decisions"
            case .emotionallyCharged: return "Emotionally Charged Moments"
            case .socialPressure: return "Social Pressure"
            case .quietReflection: return "Quiet Reflection"
            case .routineTask: return "Routine Tasks"
            case .novelSituation: return "Novel Situations"
            }
        }
    }
    
    public struct SituationalFactor: Codable, Identifiable {
        public let id: UUID
        public let factor: String
        public let impact: FactorImpact
        public let strength: Double // -1.0 to 1.0
        public let confidence: Double
        
        public init(
            id: UUID = UUID(),
            factor: String,
            impact: FactorImpact,
            strength: Double,
            confidence: Double
        ) {
            self.id = id
            self.factor = factor
            self.impact = impact
            self.strength = strength
            self.confidence = confidence
        }
    }
    
    public enum FactorImpact: String, CaseIterable, Codable {
        case enhances = "enhances"
        case neutral = "neutral"
        case impairs = "impairs"
        
        public var description: String {
            switch self {
            case .enhances: return "Enhances Performance"
            case .neutral: return "No Impact"
            case .impairs: return "Impairs Performance"
            }
        }
    }
    
    public struct SelectiveValidation: Codable {
        public let domain: AbilityDomain
        public let validated: Bool
        public let confidence: Double
        public let evidence: [PerformanceEvent]
        public let recommendedStrategy: ConfidenceStrategy
        
        public init(
            domain: AbilityDomain,
            validated: Bool,
            confidence: Double,
            evidence: [PerformanceEvent],
            recommendedStrategy: ConfidenceStrategy
        ) {
            self.domain = domain
            self.validated = validated
            self.confidence = confidence
            self.evidence = evidence
            self.recommendedStrategy = recommendedStrategy
        }
    }
    
    public struct ConfidenceStrategy: Codable {
        public let strategy: StrategyType
        public let description: String
        public let implementationSteps: [String]
        public let expectedOutcome: String
        public let riskMitigation: [String]
        
        public init(
            strategy: StrategyType,
            description: String,
            implementationSteps: [String],
            expectedOutcome: String,
            riskMitigation: [String]
        ) {
            self.strategy = strategy
            self.description = description
            self.implementationSteps = implementationSteps
            self.expectedOutcome = expectedOutcome
            self.riskMitigation = riskMitigation
        }
    }
    
    public enum StrategyType: String, CaseIterable, Codable {
        case trustWithBackup = "trust_with_backup"
        case contextualTrust = "contextual_trust"
        case gradualConfidenceBuilding = "gradual_confidence_building"
        case domainSpecificValidation = "domain_specific_validation"
        case expectationReframing = "expectation_reframing"
        
        public var description: String {
            switch self {
            case .trustWithBackup: return "Trust with Safety Net"
            case .contextualTrust: return "Context-Aware Trust"
            case .gradualConfidenceBuilding: return "Gradual Confidence Building"
            case .domainSpecificValidation: return "Domain-Specific Validation"
            case .expectationReframing: return "Expectation Reframing"
            }
        }
    }
    
    public struct PerformanceEvent: Codable, Identifiable {
        public let id: UUID
        public let domain: AbilityDomain
        public let context: ContextType
        public let accuracy: Double
        public let confidence: Double
        public let timestamp: Date
        public let description: String
        public let outcome: EventOutcome
        
        public init(
            id: UUID = UUID(),
            domain: AbilityDomain,
            context: ContextType,
            accuracy: Double,
            confidence: Double,
            timestamp: Date = Date(),
            description: String,
            outcome: EventOutcome
        ) {
            self.id = id
            self.domain = domain
            self.context = context
            self.accuracy = accuracy
            self.confidence = confidence
            self.timestamp = timestamp
            self.description = description
            self.outcome = outcome
        }
    }
    
    public enum EventOutcome: String, CaseIterable, Codable {
        case success = "success"
        case partialSuccess = "partial_success"
        case failure = "failure"
        case avoided = "avoided"
        case fallbackUsed = "fallback_used"
        
        public var description: String {
            switch self {
            case .success: return "Complete Success"
            case .partialSuccess: return "Partial Success"
            case .failure: return "Failed Attempt"
            case .avoided: return "Avoided Challenge"
            case .fallbackUsed: return "Used Backup System"
            }
        }
    }
    
    // MARK: - Properties
    
    @Published public private(set) var currentAbilityMap: AbilityMap?
    @Published public private(set) var isAnalyzing = false
    @Published public private(set) var recentValidations: [SelectiveValidation] = []
    
    private var performanceHistory: [PerformanceEvent] = []
    private let contextualTracker: ContextualPerformanceTracker
    
    // MARK: - Initialization
    
    public init() {
        self.contextualTracker = ContextualPerformanceTracker()
        
        // Load sample data for demonstration
        loadSamplePerformanceData()
    }
    
    // MARK: - Public Methods
    
    /// Map user's cognitive abilities to specific domains and contexts
    public func mapAbilityDomains(performanceHistory: [PerformanceEvent]) async -> AbilityMap {
        isAnalyzing = true
        defer { isAnalyzing = false }
        
        var domainProfiles: [AbilityDomain: PerformanceProfile] = [:]
        
        // Analyze each domain
        for domain in AbilityDomain.allCases {
            let domainEvents = performanceHistory.filter { $0.domain == domain }
            
            guard !domainEvents.isEmpty else { continue }
            
            let profile = await analyzeDomainPerformance(events: domainEvents)
            domainProfiles[domain] = profile
        }
        
        let overallConfidence = calculateOverallConfidence(profiles: domainProfiles)
        
        let abilityMap = AbilityMap(
            domains: domainProfiles,
            overallConfidence: overallConfidence
        )
        
        DispatchQueue.main.async {
            self.currentAbilityMap = abilityMap
        }
        
        return abilityMap
    }
    
    /// Detect patterns in when exceptional abilities activate
    public func identifyOptimalConditions(ability: AbilityDomain) async -> [ActivationContext] {
        let relevantEvents = performanceHistory.filter { $0.domain == ability }
        let successfulEvents = relevantEvents.filter { $0.accuracy > 0.8 }
        
        var contextCounts: [ContextType: Int] = [:]
        for event in successfulEvents {
            contextCounts[event.context, default: 0] += 1
        }
        
        return contextCounts.compactMap { (contextType, count) in
            let probability = Double(count) / Double(successfulEvents.count)
            guard probability > 0.3 else { return nil }
            
            return ActivationContext(
                contextType: contextType,
                description: generateContextDescription(for: contextType, in: ability),
                activationProbability: probability,
                requiredConditions: getRequiredConditions(for: contextType)
            )
        }.sorted { $0.activationProbability > $1.activationProbability }
    }
    
    /// Distinguish between selective excellence and general expectation
    public func validateSelectiveStrengths(
        claimed: AbilityDomain,
        evidence: [PerformanceEvent]
    ) async -> SelectiveValidation {
        let domainEvents = evidence.filter { $0.domain == claimed }
        
        guard !domainEvents.isEmpty else {
            return SelectiveValidation(
                domain: claimed,
                validated: false,
                confidence: 0.0,
                evidence: [],
                recommendedStrategy: generateRecommendedStrategy(for: claimed, validated: false)
            )
        }
        
        let averageAccuracy = domainEvents.reduce(0) { $0 + $1.accuracy } / Double(domainEvents.count)
        let validated = averageAccuracy > 0.7 && domainEvents.count >= 3
        let confidence = min(averageAccuracy, Double(domainEvents.count) / 10.0)
        
        let validation = SelectiveValidation(
            domain: claimed,
            validated: validated,
            confidence: confidence,
            evidence: domainEvents,
            recommendedStrategy: generateRecommendedStrategy(for: claimed, validated: validated)
        )
        
        DispatchQueue.main.async {
            self.recentValidations.append(validation)
        }
        
        return validation
    }
    
    /// Provide context-aware confidence building
    public func buildContextualConfidence(
        domain: AbilityDomain,
        context: ContextType
    ) async -> ConfidenceStrategy {
        let relevantEvents = performanceHistory.filter {
            $0.domain == domain && $0.context == context
        }
        
        let successRate = relevantEvents.isEmpty ? 0.0 :
            Double(relevantEvents.filter { $0.accuracy > 0.7 }.count) / Double(relevantEvents.count)
        
        if successRate > 0.8 {
            return ConfidenceStrategy(
                strategy: .contextualTrust,
                description: "You have proven excellence in this domain under these conditions",
                implementationSteps: [
                    "Acknowledge your \(Int(successRate * 100))% success rate in \(context.description.lowercased())",
                    "Trust your ability when these conditions are present",
                    "Keep minimal backup only for extreme edge cases"
                ],
                expectedOutcome: "Increased confidence and performance in optimal contexts",
                riskMitigation: ["Maintain awareness of context changes", "Have fallback ready but don't default to it"]
            )
        } else if successRate > 0.5 {
            return ConfidenceStrategy(
                strategy: .gradualConfidenceBuilding,
                description: "Build confidence gradually while maintaining safety nets",
                implementationSteps: [
                    "Start with low-stakes situations in this domain",
                    "Document successful outcomes to build evidence",
                    "Gradually increase trust as success rate improves"
                ],
                expectedOutcome: "Steady improvement in confidence and performance",
                riskMitigation: ["Always have backup systems available", "Start with reversible decisions"]
            )
        } else {
            return ConfidenceStrategy(
                strategy: .domainSpecificValidation,
                description: "Focus on your actual strengths rather than assumed weaknesses",
                implementationSteps: [
                    "Identify domains where you do excel",
                    "Accept that not all abilities are universal",
                    "Use appropriate tools for different types of tasks"
                ],
                expectedOutcome: "Realistic confidence calibration and optimal tool usage",
                riskMitigation: ["Don't over-generalize from one domain to another"]
            )
        }
    }
    
    /// Record a new performance event
    public func recordPerformanceEvent(_ event: PerformanceEvent) {
        performanceHistory.append(event)
        
        // Update ability map if significant new data
        if performanceHistory.count % 5 == 0 {
            Task {
                await mapAbilityDomains(performanceHistory: performanceHistory)
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func analyzeDomainPerformance(events: [PerformanceEvent]) async -> PerformanceProfile {
        let averageAccuracy = events.reduce(0) { $0 + $1.accuracy } / Double(events.count)
        
        let contextualFactors = await contextualTracker.analyzeSituationalFactors(performance: events)
        let optimalConditions = await identifyOptimalConditions(ability: events.first?.domain ?? .meaningfulSequences)
        
        let confidenceCalibration = calculateConfidenceCalibration(events: events)
        
        return PerformanceProfile(
            averageAccuracy: averageAccuracy,
            contextualFactors: contextualFactors,
            optimalConditions: optimalConditions,
            confidenceCalibration: confidenceCalibration,
            evidenceCount: events.count
        )
    }
    
    private func calculateOverallConfidence(profiles: [AbilityDomain: PerformanceProfile]) -> Double {
        guard !profiles.isEmpty else { return 0.0 }
        
        let weightedSum = profiles.reduce(0.0) { result, pair in
            let (_, profile) = pair
            let weight = min(1.0, Double(profile.evidenceCount) / 10.0)
            return result + (profile.averageAccuracy * weight)
        }
        
        let totalWeight = profiles.reduce(0.0) { result, pair in
            let (_, profile) = pair
            return result + min(1.0, Double(profile.evidenceCount) / 10.0)
        }
        
        return totalWeight > 0 ? weightedSum / totalWeight : 0.0
    }
    
    private func calculateConfidenceCalibration(events: [PerformanceEvent]) -> Double {
        guard !events.isEmpty else { return 0.0 }
        
        let calibrationErrors = events.map { event in
            abs(event.confidence - event.accuracy)
        }
        
        let averageError = calibrationErrors.reduce(0, +) / Double(calibrationErrors.count)
        return max(0.0, 1.0 - averageError)
    }
    
    private func generateContextDescription(for contextType: ContextType, in domain: AbilityDomain) -> String {
        switch (contextType, domain) {
        case (.highStakes, .meaningfulSequences):
            return "High-pressure situations requiring recall of important sequences"
        case (.timePressed, .pressureSituations):
            return "Quick decisions under time pressure"
        case (.emotionallyCharged, .emotionalMemories):
            return "Emotionally significant moments and memories"
        default:
            return "\(contextType.description) in \(domain.description)"
        }
    }
    
    private func getRequiredConditions(for contextType: ContextType) -> [String] {
        switch contextType {
        case .highStakes:
            return ["Clear consequences", "Time pressure", "External validation needed"]
        case .timePressed:
            return ["Limited time", "Immediate decision required"]
        case .emotionallyCharged:
            return ["Personal significance", "Emotional engagement"]
        case .socialPressure:
            return ["Others watching", "Reputation at stake"]
        case .quietReflection:
            return ["Undisturbed environment", "Time to think"]
        case .routineTask:
            return ["Familiar context", "Low stakes"]
        case .novelSituation:
            return ["New experience", "Learning opportunity"]
        }
    }
    
    private func generateRecommendedStrategy(for domain: AbilityDomain, validated: Bool) -> ConfidenceStrategy {
        if validated {
            return ConfidenceStrategy(
                strategy: .trustWithBackup,
                description: "Trust your proven ability in this domain",
                implementationSteps: [
                    "Acknowledge your validated strength in \(domain.description)",
                    "Use this ability as your primary approach",
                    "Keep backup systems minimal and secondary"
                ],
                expectedOutcome: "Increased utilization of proven strengths",
                riskMitigation: ["Monitor for context changes", "Maintain but don't over-rely on backups"]
            )
        } else {
            return ConfidenceStrategy(
                strategy: .expectationReframing,
                description: "Adjust expectations to match actual performance patterns",
                implementationSteps: [
                    "Accept that \(domain.description) may not be a strength area",
                    "Focus on domains where you do excel",
                    "Use appropriate tools and strategies for this domain"
                ],
                expectedOutcome: "Realistic confidence and optimal strategy selection",
                riskMitigation: ["Don't assume failure, just use appropriate support"]
            )
        }
    }
    
    private func loadSamplePerformanceData() {
        // Sample data based on the user's actual examples
        let sampleEvents = [
            PerformanceEvent(
                domain: .meaningfulSequences,
                context: .highStakes,
                accuracy: 0.8, // 12 out of 15 digits
                confidence: 0.3, // Low confidence but high performance
                description: "Bank number recall during phone verification",
                outcome: .partialSuccess
            ),
            PerformanceEvent(
                domain: .emotionalMemories,
                context: .emotionallyCharged,
                accuracy: 1.0, // Perfect recall
                confidence: 0.95,
                description: "First words as a child: 'No, it was my idea'",
                outcome: .success
            ),
            PerformanceEvent(
                domain: .routineInformation,
                context: .routineTask,
                accuracy: 0.2, // Low password recall
                confidence: 0.1,
                description: "Password recall without written backup",
                outcome: .fallbackUsed
            )
        ]
        
        performanceHistory = sampleEvents
        
        Task {
            await mapAbilityDomains(performanceHistory: performanceHistory)
        }
    }
}

// MARK: - Contextual Performance Tracker

public class ContextualPerformanceTracker {
    
    /// Track when abilities work well vs. when they don't
    public func trackPerformanceByContext(
        ability: AbilityDomain,
        contexts: [ContextType]
    ) async -> [ContextType: Double] {
        // Implementation would track performance across different contexts
        var performanceMap: [ContextType: Double] = [:]
        
        for context in contexts {
            // This would be based on actual performance data
            performanceMap[context] = generateContextualPerformance(for: ability, in: context)
        }
        
        return performanceMap
    }
    
    /// Identify situational factors that enhance/diminish abilities
    public func analyzeSituationalFactors(performance: [PerformanceEvent]) async -> [SituationalFactor] {
        var factors: [SituationalFactor] = []
        
        // Analyze pressure factor
        let pressureEvents = performance.filter { $0.context == .highStakes || $0.context == .timePressed }
        let normalEvents = performance.filter { $0.context != .highStakes && $0.context != .timePressed }
        
        if !pressureEvents.isEmpty && !normalEvents.isEmpty {
            let pressureAccuracy = pressureEvents.reduce(0) { $0 + $1.accuracy } / Double(pressureEvents.count)
            let normalAccuracy = normalEvents.reduce(0) { $0 + $1.accuracy } / Double(normalEvents.count)
            
            let pressureImpact = pressureAccuracy - normalAccuracy
            
            factors.append(SituationalFactor(
                factor: "High Pressure",
                impact: pressureImpact > 0.1 ? .enhances : pressureImpact < -0.1 ? .impairs : .neutral,
                strength: pressureImpact,
                confidence: min(0.9, Double(pressureEvents.count + normalEvents.count) / 10.0)
            ))
        }
        
        // Analyze emotional significance
        let emotionalEvents = performance.filter { $0.context == .emotionallyCharged }
        if !emotionalEvents.isEmpty {
            let emotionalAccuracy = emotionalEvents.reduce(0) { $0 + $1.accuracy } / Double(emotionalEvents.count)
            
            factors.append(SituationalFactor(
                factor: "Emotional Significance",
                impact: emotionalAccuracy > 0.8 ? .enhances : .neutral,
                strength: emotionalAccuracy,
                confidence: min(0.9, Double(emotionalEvents.count) / 5.0)
            ))
        }
        
        return factors
    }
    
    /// Predict optimal conditions for ability utilization
    public func predictOptimalConditions(
        ability: AbilityDomain,
        upcomingSituation: ContextType
    ) async -> Double {
        // This would use ML models to predict success probability
        return generateContextualPerformance(for: ability, in: upcomingSituation)
    }
    
    private func generateContextualPerformance(for ability: AbilityDomain, in context: ContextType) -> Double {
        // Sample implementation based on the patterns we've identified
        switch (ability, context) {
        case (.meaningfulSequences, .highStakes):
            return 0.85 // High performance under pressure
        case (.emotionalMemories, .emotionallyCharged):
            return 0.95 // Excellent emotional memory
        case (.routineInformation, .routineTask):
            return 0.25 // Poor routine information recall
        case (.pressureSituations, .timePressed):
            return 0.8 // Good under time pressure
        default:
            return 0.5 // Average performance
        }
    }
} 