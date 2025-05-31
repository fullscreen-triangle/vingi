import Foundation
import Combine

/// Comprehensive cognitive pattern detector that identifies all four major cognitive inefficiency patterns
public class CognitivePatternDetector: ObservableObject {
    
    // MARK: - Properties
    
    @Published public private(set) var detectedPatterns: [CognitivePattern] = []
    @Published public private(set) var currentCognitiveLoad: CognitiveLoadAssessment?
    @Published public private(set) var isAnalyzing = false
    
    private let behaviorAnalyzer: BehaviorPatternAnalyzer
    private let paralysisDetector: AnalysisParalysisDetector
    private let tunnelVisionMonitor: TunnelVisionMonitor
    private let selectiveAbilityEngine: SelectiveAbilityRecognitionEngine
    
    // MARK: - Initialization
    
    public init() {
        self.behaviorAnalyzer = BehaviorPatternAnalyzer()
        self.paralysisDetector = AnalysisParalysisDetector()
        self.tunnelVisionMonitor = TunnelVisionMonitor()
        self.selectiveAbilityEngine = SelectiveAbilityRecognitionEngine()
    }
    
    // MARK: - Public Methods
    
    /// Detect all active cognitive patterns from user behavior
    public func detectActivePatterns(userBehavior: UserBehaviorData) async -> [CognitivePattern] {
        isAnalyzing = true
        defer { isAnalyzing = false }
        
        var patterns: [CognitivePattern] = []
        
        // Detect analysis paralysis
        if let paralysisPattern = await detectAnalysisParalysis(userBehavior.recentDecisions) {
            patterns.append(paralysisPattern)
        }
        
        // Detect tunnel vision planning
        if let tunnelVisionPattern = await detectTunnelVision(userBehavior.planningActivity) {
            patterns.append(tunnelVisionPattern)
        }
        
        // Detect default behavior loops
        let defaultLoops = await detectDefaultLoops(userBehavior.routineChoices)
        patterns.append(contentsOf: defaultLoops)
        
        // Detect exceptional ability self-doubt
        if let abilityDoubtPattern = await detectExceptionalAbilitySelfDoubt(userBehavior.performanceHistory) {
            patterns.append(abilityDoubtPattern)
        }
        
        DispatchQueue.main.async {
            self.detectedPatterns = patterns
        }
        
        return patterns
    }
    
    /// Assess overall cognitive load and pattern risks
    public func assessCognitiveLoad(userContext: UserContext) async -> CognitiveLoadAssessment {
        isAnalyzing = true
        defer { isAnalyzing = false }
        
        // Base complexity factors
        let taskComplexity = assessTaskComplexity(userContext.currentTask)
        let decisionCount = userContext.recentDecisionCount
        let contextSwitches = userContext.contextSwitchCount
        
        // Pattern-specific risk factors
        let paralysisRisk = await detectDecisionLoops(userContext.researchTime)
        let tunnelRisk = await assessDomainFocusIntensity(userContext.planningActivity)
        let defaultRisk = measureChoicePatternRigidity(userContext.routineChoices)
        let abilityDoubtRisk = await assessAbilityConfidenceGap(userContext.performanceHistory)
        
        let overallLoad = calculateCompositeLoad(
            taskComplexity: taskComplexity,
            decisionCount: decisionCount,
            contextSwitches: contextSwitches
        )
        
        let patternRisks = [
            "analysis_paralysis": paralysisRisk,
            "tunnel_vision": tunnelRisk,
            "default_loops": defaultRisk,
            "exceptional_ability_self_doubt": abilityDoubtRisk
        ]
        
        let interventions = generateInterventions(
            paralysisRisk: paralysisRisk,
            tunnelRisk: tunnelRisk,
            defaultRisk: defaultRisk,
            abilityDoubtRisk: abilityDoubtRisk
        )
        
        let urgencyLevel = determineUrgencyLevel(
            overallLoad: overallLoad,
            maxRisk: patternRisks.values.max() ?? 0.0
        )
        
        let assessment = CognitiveLoadAssessment(
            overallLoad: overallLoad,
            patternRisks: patternRisks,
            recommendedInterventions: interventions,
            urgencyLevel: urgencyLevel
        )
        
        DispatchQueue.main.async {
            self.currentCognitiveLoad = assessment
        }
        
        return assessment
    }
    
    /// Predict pattern emergence before manifestation
    public func predictPatternRisk(behaviorTrends: [UserBehaviorData]) async -> [String: Double] {
        var riskPredictions: [String: Double] = [:]
        
        // Analyze trends for each pattern type
        riskPredictions["analysis_paralysis"] = await predictParalysisRisk(behaviorTrends)
        riskPredictions["tunnel_vision"] = await predictTunnelVisionRisk(behaviorTrends)
        riskPredictions["default_loops"] = await predictDefaultLoopRisk(behaviorTrends)
        riskPredictions["exceptional_ability_self_doubt"] = await predictAbilityDoubtRisk(behaviorTrends)
        
        return riskPredictions
    }
    
    // MARK: - Pattern Detection Methods
    
    private func detectAnalysisParalysis(_ decisions: [DecisionEvent]) async -> CognitivePattern? {
        let recentDecisions = decisions.filter { 
            Date().timeIntervalSince($0.timestamp) < 3600 // Last hour
        }
        
        guard !recentDecisions.isEmpty else { return nil }
        
        // Check for paralysis indicators
        let avgResearchTime = recentDecisions.reduce(0) { $0 + $1.researchTime } / Double(recentDecisions.count)
        let avgOptionsConsidered = recentDecisions.reduce(0) { $0 + $1.optionsConsidered } / recentDecisions.count
        let indecisionRate = Double(recentDecisions.filter { !$0.decisionMade }.count) / Double(recentDecisions.count)
        
        // Paralysis thresholds
        let isParalyzed = avgResearchTime > 1200 || // 20+ minutes
                         avgOptionsConsidered > 10 ||
                         indecisionRate > 0.5
        
        guard isParalyzed else { return nil }
        
        let severity = determineSeverity(
            researchTime: avgResearchTime,
            optionsCount: avgOptionsConsidered,
            indecisionRate: indecisionRate
        )
        
        let domain = recentDecisions.first?.domain ?? "unknown"
        let interventions = generateParalysisInterventions(severity: severity)
        
        return .analysisParalysis(
            severity: severity,
            triggerDomain: domain,
            suggestedInterventions: interventions
        )
    }
    
    private func detectTunnelVision(_ planningEvents: [PlanningEvent]) async -> CognitivePattern? {
        let recentPlanning = planningEvents.filter {
            Date().timeIntervalSince($0.timestamp) < 86400 // Last 24 hours
        }
        
        guard recentPlanning.count >= 2 else { return nil }
        
        // Calculate domain focus distribution
        var domainTimeSpent: [String: TimeInterval] = [:]
        for event in recentPlanning {
            domainTimeSpent[event.domain, default: 0] += event.timeSpent
        }
        
        let totalTime = domainTimeSpent.values.reduce(0, +)
        let maxDomainTime = domainTimeSpent.values.max() ?? 0
        let focusRatio = maxDomainTime / totalTime
        
        // Tunnel vision threshold
        guard focusRatio > 0.8 else { return nil }
        
        let focusDomain = domainTimeSpent.max(by: { $0.value < $1.value })?.key ?? "unknown"
        let neglectedDomains = domainTimeSpent.filter { $0.value / totalTime < 0.05 }.map { $0.key }
        
        let riskLevel = determineTunnelRisk(
            focusRatio: focusRatio,
            neglectedDomainCount: neglectedDomains.count
        )
        
        return .tunnelVision(
            focusDomain: focusDomain,
            neglectedDomains: neglectedDomains,
            riskLevel: riskLevel
        )
    }
    
    private func detectDefaultLoops(_ choiceEvents: [ChoiceEvent]) async -> [CognitivePattern] {
        var patterns: [CognitivePattern] = []
        
        // Group choices by domain
        let choicesByDomain = Dictionary(grouping: choiceEvents) { $0.domain }
        
        for (domain, choices) in choicesByDomain {
            guard choices.count >= 5 else { continue } // Need minimum history
            
            // Calculate choice repetition rate
            let uniqueChoices = Set(choices.map { $0.choice })
            let repetitionRate = 1.0 - (Double(uniqueChoices.count) / Double(choices.count))
            
            // Default loop threshold
            guard repetitionRate > 0.8 else { continue }
            
            // Identify constraint type
            let constraintType = identifyConstraintType(choices: choices)
            
            // Calculate optimization potential
            let optimizationPotential = calculateOptimizationPotential(
                choices: choices,
                constraintType: constraintType
            )
            
            patterns.append(.defaultLoop(
                domain: domain,
                constraintType: constraintType,
                optimizationPotential: optimizationPotential
            ))
        }
        
        return patterns
    }
    
    private func detectExceptionalAbilitySelfDoubt(_ performanceHistory: [SelectiveAbilityRecognitionEngine.PerformanceEvent]) async -> CognitivePattern? {
        guard performanceHistory.count >= 3 else { return nil }
        
        // Group by ability domain
        let eventsByDomain = Dictionary(grouping: performanceHistory) { $0.domain }
        
        for (domain, events) in eventsByDomain {
            guard events.count >= 3 else { continue }
            
            // Calculate actual vs perceived performance
            let actualPerformance = events.reduce(0) { $0 + $1.accuracy } / Double(events.count)
            let perceivedPerformance = events.reduce(0) { $0 + $1.confidence } / Double(events.count)
            
            // Detect significant confidence gap
            let confidenceGap = actualPerformance - perceivedPerformance
            
            // Self-doubt threshold: high actual performance, low confidence
            guard actualPerformance > 0.7 && confidenceGap > 0.3 else { continue }
            
            return .exceptionalAbilitySelfDoubt(
                ability: domain,
                evidenceCount: events.count,
                actualPerformance: actualPerformance,
                perceivedPerformance: perceivedPerformance,
                confidenceGap: confidenceGap
            )
        }
        
        return nil
    }
    
    // MARK: - Helper Methods
    
    private func determineSeverity(researchTime: TimeInterval, optionsCount: Int, indecisionRate: Double) -> ParalysisSeverity {
        let score = (researchTime / 1800.0) + (Double(optionsCount) / 15.0) + indecisionRate
        
        switch score {
        case 0..<1.0: return .mild
        case 1.0..<2.0: return .moderate
        case 2.0..<3.0: return .severe
        default: return .critical
        }
    }
    
    private func determineTunnelRisk(focusRatio: Double, neglectedDomainCount: Int) -> TunnelRisk {
        switch (focusRatio, neglectedDomainCount) {
        case (0.8..<0.9, 1...2): return .low
        case (0.9..<0.95, 1...3): return .medium
        case (0.95..., _): return .critical
        default: return .high
        }
    }
    
    private func identifyConstraintType(choices: [ChoiceEvent]) -> ConstraintType {
        // Analyze choice patterns to identify primary constraint
        let locations = Set(choices.compactMap { $0.metadata["location"] })
        let venues = Set(choices.compactMap { $0.metadata["venue"] })
        
        if locations.count == 1 {
            return .location
        } else if venues.count <= 2 {
            return .convenience
        } else {
            return .familiarity
        }
    }
    
    private func calculateOptimizationPotential(choices: [ChoiceEvent], constraintType: ConstraintType) -> Double {
        // Calculate how much improvement is possible
        switch constraintType {
        case .location: return 0.4 // Shopping route optimization
        case .convenience: return 0.6 // Time vs quality tradeoff
        case .familiarity: return 0.8 // Exploration potential
        default: return 0.3
        }
    }
    
    private func generateParalysisInterventions(severity: ParalysisSeverity) -> [String] {
        switch severity {
        case .mild:
            return ["Set 10-minute decision timer", "Use 'good enough' principle"]
        case .moderate:
            return ["Limit to 3 options max", "Set decision deadline", "Use decision matrix"]
        case .severe:
            return ["Immediate time boxing", "Delegate decision if possible", "Choose first viable option"]
        case .critical:
            return ["Emergency decision protocol", "Choose any reasonable option immediately"]
        }
    }
    
    private func assessTaskComplexity(_ task: String?) -> Double {
        // Simplified complexity assessment
        guard let task = task else { return 0.3 }
        
        let keywords = ["plan", "research", "compare", "analyze", "optimize"]
        let complexityIndicators = keywords.filter { task.lowercased().contains($0) }.count
        
        return min(1.0, Double(complexityIndicators) / 3.0)
    }
    
    private func calculateCompositeLoad(taskComplexity: Double, decisionCount: Int, contextSwitches: Int) -> Double {
        let baseLoad = taskComplexity
        let switchPenalty = pow(1.23, Double(contextSwitches))
        let decisionFatigue = min(2.0, 1 + (Double(decisionCount) / 10.0))
        
        return min(1.0, baseLoad * switchPenalty * decisionFatigue / 10.0)
    }
    
    private func determineUrgencyLevel(overallLoad: Double, maxRisk: Double) -> UrgencyLevel {
        let combinedScore = (overallLoad + maxRisk) / 2.0
        
        switch combinedScore {
        case 0..<0.3: return .monitoring
        case 0.3..<0.6: return .attention
        case 0.6..<0.8: return .intervention
        default: return .emergency
        }
    }
    
    private func generateInterventions(paralysisRisk: Double, tunnelRisk: Double, defaultRisk: Double, abilityDoubtRisk: Double) -> [String] {
        var interventions: [String] = []
        
        if paralysisRisk > 0.6 {
            interventions.append("Apply time boxing to decisions")
        }
        
        if tunnelRisk > 0.6 {
            interventions.append("Review all planning domains")
        }
        
        if defaultRisk > 0.6 {
            interventions.append("Explore safe alternatives")
        }
        
        if abilityDoubtRisk > 0.6 {
            interventions.append("Trust validated abilities")
        }
        
        return interventions
    }
    
    // MARK: - Risk Assessment Methods
    
    private func detectDecisionLoops(_ researchTime: TimeInterval?) async -> Double {
        guard let researchTime = researchTime else { return 0.0 }
        return min(1.0, researchTime / 3600.0) // 1 hour = max risk
    }
    
    private func assessDomainFocusIntensity(_ planningActivity: [PlanningEvent]) async -> Double {
        guard !planningActivity.isEmpty else { return 0.0 }
        
        let domainCounts = Dictionary(grouping: planningActivity) { $0.domain }.mapValues { $0.count }
        let maxCount = domainCounts.values.max() ?? 0
        let totalCount = planningActivity.count
        
        return Double(maxCount) / Double(totalCount)
    }
    
    private func measureChoicePatternRigidity(_ routineChoices: [ChoiceEvent]) -> Double {
        guard routineChoices.count >= 5 else { return 0.0 }
        
        let uniqueChoices = Set(routineChoices.map { $0.choice })
        let repetitionRate = 1.0 - (Double(uniqueChoices.count) / Double(routineChoices.count))
        
        return repetitionRate
    }
    
    private func assessAbilityConfidenceGap(_ performanceHistory: [SelectiveAbilityRecognitionEngine.PerformanceEvent]) async -> Double {
        guard !performanceHistory.isEmpty else { return 0.0 }
        
        let avgActual = performanceHistory.reduce(0) { $0 + $1.accuracy } / Double(performanceHistory.count)
        let avgConfidence = performanceHistory.reduce(0) { $0 + $1.confidence } / Double(performanceHistory.count)
        
        return max(0.0, avgActual - avgConfidence)
    }
    
    // MARK: - Prediction Methods
    
    private func predictParalysisRisk(_ trends: [UserBehaviorData]) async -> Double {
        let recentResearchTimes = trends.flatMap { $0.recentDecisions.map { $0.researchTime } }
        guard !recentResearchTimes.isEmpty else { return 0.0 }
        
        let avgResearchTime = recentResearchTimes.reduce(0, +) / Double(recentResearchTimes.count)
        return min(1.0, avgResearchTime / 1800.0) // 30 minutes = max risk
    }
    
    private func predictTunnelVisionRisk(_ trends: [UserBehaviorData]) async -> Double {
        let recentPlanning = trends.flatMap { $0.planningActivity }
        return await assessDomainFocusIntensity(recentPlanning)
    }
    
    private func predictDefaultLoopRisk(_ trends: [UserBehaviorData]) async -> Double {
        let recentChoices = trends.flatMap { $0.routineChoices }
        return measureChoicePatternRigidity(recentChoices)
    }
    
    private func predictAbilityDoubtRisk(_ trends: [UserBehaviorData]) async -> Double {
        let recentPerformance = trends.flatMap { $0.performanceHistory }
        return await assessAbilityConfidenceGap(recentPerformance)
    }
}

// MARK: - Supporting Classes

public class BehaviorPatternAnalyzer {
    public func identifyDefaultLoops(_ choices: [ChoiceEvent]) -> [DefaultLoop] {
        // Implementation for identifying default behavior patterns
        return []
    }
}

public class AnalysisParalysisDetector {
    public func isInParalysisLoop(_ decisions: [DecisionEvent]) -> Bool {
        // Implementation for detecting paralysis
        return false
    }
    
    public func assessSeverity() -> ParalysisSeverity {
        return .mild
    }
    
    public func identifyDomain() -> String {
        return "unknown"
    }
    
    public func generateInterventions() -> [String] {
        return []
    }
}

public class TunnelVisionMonitor {
    public func detectsTunnelVision(_ planning: [PlanningEvent]) -> Bool {
        // Implementation for detecting tunnel vision
        return false
    }
    
    public func getPrimaryFocus() -> String {
        return "unknown"
    }
    
    public func getNeglectedDomains() -> [String] {
        return []
    }
    
    public func assessRisk() -> TunnelRisk {
        return .low
    }
}

// MARK: - Supporting Types

public struct DefaultLoop {
    public let domain: String
    public let constraintType: ConstraintType
    
    public func calculateOptimizationPotential() -> Double {
        return 0.5
    }
}

public struct UserContext {
    public let currentTask: String?
    public let recentDecisionCount: Int
    public let contextSwitchCount: Int
    public let researchTime: TimeInterval?
    public let planningActivity: [PlanningEvent]
    public let routineChoices: [ChoiceEvent]
    public let performanceHistory: [SelectiveAbilityRecognitionEngine.PerformanceEvent]
    
    public init(
        currentTask: String? = nil,
        recentDecisionCount: Int = 0,
        contextSwitchCount: Int = 0,
        researchTime: TimeInterval? = nil,
        planningActivity: [PlanningEvent] = [],
        routineChoices: [ChoiceEvent] = [],
        performanceHistory: [SelectiveAbilityRecognitionEngine.PerformanceEvent] = []
    ) {
        self.currentTask = currentTask
        self.recentDecisionCount = recentDecisionCount
        self.contextSwitchCount = contextSwitchCount
        self.researchTime = researchTime
        self.planningActivity = planningActivity
        self.routineChoices = routineChoices
        self.performanceHistory = performanceHistory
    }
} 