import Foundation
import Combine

/// Engine that provides intelligent defaults and prevents endless decision loops
/// Core principle: "Good enough" decisions made quickly are better than perfect decisions made slowly
public class DecisionEngine: ObservableObject {
    
    // MARK: - Types
    
    public struct Decision: Identifiable, Codable {
        public let id: UUID
        public let question: String
        public let options: [DecisionOption]
        public let recommendedChoice: UUID
        public let confidence: Double
        public let reasoning: String
        public let timeoutSeconds: TimeInterval
        public let category: DecisionCategory
        
        public init(
            id: UUID = UUID(),
            question: String,
            options: [DecisionOption],
            recommendedChoice: UUID,
            confidence: Double,
            reasoning: String,
            timeoutSeconds: TimeInterval = 300, // 5 minutes default
            category: DecisionCategory
        ) {
            self.id = id
            self.question = question
            self.options = options
            self.recommendedChoice = recommendedChoice
            self.confidence = confidence
            self.reasoning = reasoning
            self.timeoutSeconds = timeoutSeconds
            self.category = category
        }
    }
    
    public struct DecisionOption: Identifiable, Codable {
        public let id: UUID
        public let title: String
        public let description: String
        public let score: Double // 0-1, higher is better
        public let tradeoffs: [String]
        public let estimatedCost: Double?
        public let estimatedTime: TimeInterval?
        public let riskLevel: RiskLevel
        
        public init(
            id: UUID = UUID(),
            title: String,
            description: String,
            score: Double,
            tradeoffs: [String] = [],
            estimatedCost: Double? = nil,
            estimatedTime: TimeInterval? = nil,
            riskLevel: RiskLevel = .medium
        ) {
            self.id = id
            self.title = title
            self.description = description
            self.score = score
            self.tradeoffs = tradeoffs
            self.estimatedCost = estimatedCost
            self.estimatedTime = estimatedTime
            self.riskLevel = riskLevel
        }
    }
    
    public enum DecisionCategory: String, CaseIterable, Codable {
        case travel = "travel"
        case purchase = "purchase"
        case scheduling = "scheduling"
        case communication = "communication"
        case research = "research"
        case general = "general"
    }
    
    public enum RiskLevel: String, CaseIterable, Codable {
        case veryLow = "very_low"
        case low = "low"
        case medium = "medium"
        case high = "high"
        case veryHigh = "very_high"
    }
    
    public struct DecisionResult: Codable {
        public let decision: Decision
        public let chosenOption: DecisionOption
        public let decisionMethod: DecisionMethod
        public let timeSpent: TimeInterval
        public let userSatisfaction: Double?
        
        public init(
            decision: Decision,
            chosenOption: DecisionOption,
            decisionMethod: DecisionMethod,
            timeSpent: TimeInterval,
            userSatisfaction: Double? = nil
        ) {
            self.decision = decision
            self.chosenOption = chosenOption
            self.decisionMethod = decisionMethod
            self.timeSpent = timeSpent
            self.userSatisfaction = userSatisfaction
        }
    }
    
    public enum DecisionMethod: String, CaseIterable, Codable {
        case userChoice = "user_choice"
        case intelligentDefault = "intelligent_default"
        case timeout = "timeout"
        case goodEnoughRule = "good_enough_rule"
        case previousPattern = "previous_pattern"
    }
    
    // MARK: - Properties
    
    @Published public private(set) var pendingDecisions: [Decision] = []
    @Published public private(set) var isProcessing = false
    
    private var decisionHistory: [DecisionResult] = []
    private var userPreferences: [String: Any] = [:]
    
    // Decision-making rules
    private let goodEnoughThreshold: Double = 0.7
    private let maxDecisionTime: TimeInterval = 900 // 15 minutes max
    private let quickDecisionTime: TimeInterval = 300 // 5 minutes for quick decisions
    
    // MARK: - Public Methods
    
    /// Present a decision and get intelligent recommendation or automatic choice
    public func presentDecision(_ question: String, options: [DecisionOption], category: DecisionCategory, allowAutoDecision: Bool = true) async -> DecisionResult {
        
        let startTime = Date()
        
        // Create decision with intelligent recommendation
        let decision = createDecision(question: question, options: options, category: category)
        
        DispatchQueue.main.async {
            self.pendingDecisions.append(decision)
        }
        
        // Check if we can make an automatic decision
        if allowAutoDecision && shouldAutoDecide(decision) {
            let result = makeAutomaticDecision(decision)
            
            DispatchQueue.main.async {
                self.pendingDecisions.removeAll { $0.id == decision.id }
            }
            
            recordDecision(result)
            return result
        }
        
        // Wait for user decision or timeout
        return await waitForDecision(decision, startTime: startTime)
    }
    
    /// Specifically for travel route selection - your train booking scenario
    public func selectTravelRoute(routes: [TravelRoute], preferences: TravelPreferences = TravelPreferences()) async -> DecisionResult {
        
        let options = routes.map { route in
            DecisionOption(
                title: route.description,
                description: "Duration: \(route.duration), Cost: €\(route.cost), Changes: \(route.changes)",
                score: scoreTravelRoute(route, preferences: preferences),
                tradeoffs: route.tradeoffs,
                estimatedCost: route.cost,
                estimatedTime: route.duration,
                riskLevel: route.changes > 2 ? .medium : .low
            )
        }
        
        return await presentDecision(
            "Select travel route",
            options: options,
            category: .travel,
            allowAutoDecision: true
        )
    }
    
    /// Use this when you want Vingi to just pick something reasonable
    public func makeGoodEnoughDecision(for decision: Decision) -> DecisionResult {
        let goodEnoughOptions = decision.options.filter { $0.score >= goodEnoughThreshold }
        
        let chosenOption: DecisionOption
        let method: DecisionMethod
        
        if !goodEnoughOptions.isEmpty {
            // Pick the first "good enough" option to avoid over-optimization
            chosenOption = goodEnoughOptions.first!
            method = .goodEnoughRule
        } else {
            // If no option meets the threshold, pick the best available
            chosenOption = decision.options.max(by: { $0.score < $1.score })!
            method = .intelligentDefault
        }
        
        let result = DecisionResult(
            decision: decision,
            chosenOption: chosenOption,
            decisionMethod: method,
            timeSpent: 0, // Instant decision
            userSatisfaction: nil
        )
        
        recordDecision(result)
        return result
    }
    
    // MARK: - Private Methods
    
    private func createDecision(question: String, options: [DecisionOption], category: DecisionCategory) -> Decision {
        
        let recommendedChoice = findRecommendedOption(options: options, category: category)
        let confidence = calculateConfidence(options: options, recommended: recommendedChoice)
        let reasoning = generateReasoning(options: options, recommended: recommendedChoice, category: category)
        let timeout = determineTimeout(for: category, options: options)
        
        return Decision(
            question: question,
            options: options,
            recommendedChoice: recommendedChoice,
            confidence: confidence,
            reasoning: reasoning,
            timeoutSeconds: timeout,
            category: category
        )
    }
    
    private func findRecommendedOption(options: [DecisionOption], category: DecisionCategory) -> UUID {
        // Use different strategies based on category and user history
        
        switch category {
        case .travel:
            return recommendTravelOption(options)
        case .purchase:
            return recommendPurchaseOption(options)
        default:
            // Default: pick highest scoring option, but prefer "good enough" to avoid perfectionism
            let goodEnoughOptions = options.filter { $0.score >= goodEnoughThreshold }
            if !goodEnoughOptions.isEmpty {
                // Among good enough options, pick one with lower cost/time if available
                return goodEnoughOptions.min { lhs, rhs in
                    let lhsCost = (lhs.estimatedCost ?? 0) + (lhs.estimatedTime ?? 0) / 3600
                    let rhsCost = (rhs.estimatedCost ?? 0) + (rhs.estimatedTime ?? 0) / 3600
                    return lhsCost < rhsCost
                }?.id ?? goodEnoughOptions.first!.id
            } else {
                return options.max { $0.score < $1.score }?.id ?? options.first!.id
            }
        }
    }
    
    private func recommendTravelOption(_ options: [DecisionOption]) -> UUID {
        // For travel: prefer reasonable balance of time, cost, and simplicity
        // This is specifically for your train booking scenario
        
        let scoredOptions = options.map { option in
            var score = option.score
            
            // Prefer options with fewer changes (less stress)
            if option.description.contains("Changes: 0") || option.description.contains("Direct") {
                score += 0.2
            } else if option.description.contains("Changes: 1") {
                score += 0.1
            }
            
            // Slightly penalize very expensive options unless they're significantly better
            if let cost = option.estimatedCost, cost > 50 {
                score -= 0.1
            }
            
            // Prefer "reasonable" times over very early/very late
            // This would normally check actual times, simplified here
            if option.description.lowercased().contains("reasonable") {
                score += 0.1
            }
            
            return (option.id, score)
        }
        
        return scoredOptions.max { $0.1 < $1.1 }?.0 ?? options.first!.id
    }
    
    private func recommendPurchaseOption(_ options: [DecisionOption]) -> UUID {
        // For purchases: prefer value for money, avoiding both cheapest and most expensive
        // unless there's a clear winner
        
        guard let maxCost = options.compactMap({ $0.estimatedCost }).max(),
              let minCost = options.compactMap({ $0.estimatedCost }).min() else {
            return options.max { $0.score < $1.score }?.id ?? options.first!.id
        }
        
        let costRange = maxCost - minCost
        
        let scoredOptions = options.map { option in
            var score = option.score
            
            if let cost = option.estimatedCost {
                // Prefer middle-range options (avoid extremes)
                let costPosition = (cost - minCost) / costRange
                if costPosition > 0.2 && costPosition < 0.8 {
                    score += 0.1
                }
            }
            
            return (option.id, score)
        }
        
        return scoredOptions.max { $0.1 < $1.1 }?.0 ?? options.first!.id
    }
    
    private func shouldAutoDecide(_ decision: Decision) -> Bool {
        // Auto-decide when:
        // 1. High confidence in recommendation
        // 2. Low-risk decision
        // 3. User has patterns suggesting they'd prefer automation
        // 4. Decision category is suitable for automation
        
        let highConfidence = decision.confidence > 0.8
        let lowRisk = decision.options.allSatisfy { $0.riskLevel == .low || $0.riskLevel == .veryLow }
        let suitableCategory = [.travel, .scheduling].contains(decision.category)
        let userPrefersAutomation = userPreferences["auto_decide_\(decision.category.rawValue)"] as? Bool ?? false
        
        return highConfidence && lowRisk && (suitableCategory || userPrefersAutomation)
    }
    
    private func makeAutomaticDecision(_ decision: Decision) -> DecisionResult {
        let recommendedOption = decision.options.first { $0.id == decision.recommendedChoice }!
        
        return DecisionResult(
            decision: decision,
            chosenOption: recommendedOption,
            decisionMethod: .intelligentDefault,
            timeSpent: 0,
            userSatisfaction: nil
        )
    }
    
    private func waitForDecision(_ decision: Decision, startTime: Date) async -> DecisionResult {
        // This would normally wait for user input or timeout
        // For now, simulate timeout and use intelligent default
        
        try? await Task.sleep(nanoseconds: UInt64(decision.timeoutSeconds * 1_000_000_000))
        
        let timeSpent = Date().timeIntervalSince(startTime)
        let recommendedOption = decision.options.first { $0.id == decision.recommendedChoice }!
        
        DispatchQueue.main.async {
            self.pendingDecisions.removeAll { $0.id == decision.id }
        }
        
        let result = DecisionResult(
            decision: decision,
            chosenOption: recommendedOption,
            decisionMethod: .timeout,
            timeSpent: timeSpent,
            userSatisfaction: nil
        )
        
        recordDecision(result)
        return result
    }
    
    private func calculateConfidence(options: [DecisionOption], recommended: UUID) -> Double {
        guard let recommendedOption = options.first(where: { $0.id == recommended }) else {
            return 0.5
        }
        
        let otherOptions = options.filter { $0.id != recommended }
        let maxOtherScore = otherOptions.map { $0.score }.max() ?? 0
        
        // Confidence is higher when recommended option significantly outperforms others
        let scoreDifference = recommendedOption.score - maxOtherScore
        return min(1.0, 0.5 + scoreDifference)
    }
    
    private func generateReasoning(options: [DecisionOption], recommended: UUID, category: DecisionCategory) -> String {
        guard let recommendedOption = options.first(where: { $0.id == recommended }) else {
            return "Selected based on available options."
        }
        
        switch category {
        case .travel:
            return "Recommended for good balance of time, cost, and convenience. \(recommendedOption.description)"
        case .purchase:
            return "Best value for money considering quality and price. \(recommendedOption.description)"
        default:
            return "Highest scoring option that meets your typical preferences. \(recommendedOption.description)"
        }
    }
    
    private func determineTimeout(for category: DecisionCategory, options: [DecisionOption]) -> TimeInterval {
        // Different categories get different time limits
        switch category {
        case .travel:
            return 300 // 5 minutes for travel decisions
        case .purchase:
            let maxCost = options.compactMap { $0.estimatedCost }.max() ?? 0
            return maxCost > 100 ? 600 : 300 // More time for expensive purchases
        default:
            return quickDecisionTime
        }
    }
    
    private func scoreTravelRoute(_ route: TravelRoute, preferences: TravelPreferences) -> Double {
        var score = 0.5 // Base score
        
        // Score based on duration (prefer reasonable times)
        if route.duration < 3600 { // Under 1 hour
            score += 0.3
        } else if route.duration < 7200 { // Under 2 hours
            score += 0.2
        }
        
        // Score based on cost (prefer reasonable prices)
        if route.cost < 30 {
            score += 0.2
        } else if route.cost < 60 {
            score += 0.1
        }
        
        // Score based on changes (prefer fewer)
        if route.changes == 0 {
            score += 0.2
        } else if route.changes == 1 {
            score += 0.1
        } else if route.changes > 2 {
            score -= 0.1
        }
        
        // Apply user preferences
        if preferences.preferFast && route.duration < 3600 {
            score += 0.1
        }
        
        if preferences.preferCheap && route.cost < 40 {
            score += 0.1
        }
        
        if preferences.preferDirect && route.changes == 0 {
            score += 0.15
        }
        
        return min(1.0, max(0.0, score))
    }
    
    private func recordDecision(_ result: DecisionResult) {
        decisionHistory.append(result)
        
        // Learn from decisions to improve future recommendations
        updatePreferences(from: result)
    }
    
    private func updatePreferences(from result: DecisionResult) {
        // Update user preferences based on decision patterns
        let category = result.decision.category.rawValue
        
        // If user consistently accepts automatic decisions, enable more automation
        if result.decisionMethod == .intelligentDefault {
            let currentPref = userPreferences["auto_decide_\(category)"] as? Bool ?? false
            if !currentPref {
                let recentAutoDecisions = decisionHistory.suffix(5).filter {
                    $0.decision.category == result.decision.category &&
                    $0.decisionMethod == .intelligentDefault
                }
                
                if recentAutoDecisions.count >= 3 {
                    userPreferences["auto_decide_\(category)"] = true
                }
            }
        }
    }
}

// MARK: - Supporting Types

public struct TravelRoute: Codable {
    public let description: String
    public let duration: TimeInterval // in seconds
    public let cost: Double
    public let changes: Int
    public let tradeoffs: [String]
    
    public init(description: String, duration: TimeInterval, cost: Double, changes: Int, tradeoffs: [String] = []) {
        self.description = description
        self.duration = duration
        self.cost = cost
        self.changes = changes
        self.tradeoffs = tradeoffs
    }
}

public struct TravelPreferences: Codable {
    public let preferFast: Bool
    public let preferCheap: Bool
    public let preferDirect: Bool
    public let maxCost: Double?
    public let maxDuration: TimeInterval?
    
    public init(
        preferFast: Bool = false,
        preferCheap: Bool = false,
        preferDirect: Bool = true,
        maxCost: Double? = nil,
        maxDuration: TimeInterval? = nil
    ) {
        self.preferFast = preferFast
        self.preferCheap = preferCheap
        self.preferDirect = preferDirect
        self.maxCost = maxCost
        self.maxDuration = maxDuration
    }
} 