import Foundation
import Combine

/// Core engine that breaks down complex tasks to prevent analysis paralysis
/// Specifically designed to handle the "I want to do X but get overwhelmed by planning" pattern
public class TaskBreakdownEngine: ObservableObject {
    
    // MARK: - Types
    
    public struct Task: Identifiable, Codable {
        public let id: UUID
        public let title: String
        public let description: String
        public let complexity: TaskComplexity
        public let estimatedTime: TimeInterval
        public let dependencies: [UUID]
        public let isBlocking: Bool
        public let suggestedApproach: TaskApproach
        public let automationLevel: AutomationLevel
        
        public init(
            id: UUID = UUID(),
            title: String,
            description: String,
            complexity: TaskComplexity,
            estimatedTime: TimeInterval,
            dependencies: [UUID] = [],
            isBlocking: Bool = false,
            suggestedApproach: TaskApproach,
            automationLevel: AutomationLevel
        ) {
            self.id = id
            self.title = title
            self.description = description
            self.complexity = complexity
            self.estimatedTime = estimatedTime
            self.dependencies = dependencies
            self.isBlocking = isBlocking
            self.suggestedApproach = suggestedApproach
            self.automationLevel = automationLevel
        }
    }
    
    public enum TaskComplexity: String, CaseIterable, Codable {
        case trivial    // < 5 minutes
        case simple     // 5-15 minutes
        case moderate   // 15-45 minutes
        case complex    // 45+ minutes
        case overwhelming // Needs further breakdown
    }
    
    public enum TaskApproach: String, CaseIterable, Codable {
        case doNow          // Just do it immediately
        case useDefault     // Use intelligent default option
        case simpleResearch // Quick 5-minute research, then decide
        case delegate       // Let Vingi handle it automatically
        case timeBox        // Set a timer and work within constraints
        case goodEnough     // Accept "good enough" solution
    }
    
    public enum AutomationLevel: String, CaseIterable, Codable {
        case manual         // User does everything
        case assisted       // Vingi provides information and suggestions
        case guided         // Vingi walks through step-by-step
        case automatic      // Vingi handles entirely
    }
    
    public struct BreakdownResult: Codable {
        public let originalGoal: String
        public let subtasks: [Task]
        public let recommendedNext: UUID?
        public let estimatedTotal: TimeInterval
        public let cognitiveLoadScore: Double // 0-1, lower is better
        public let paralysisRisk: ParalysisRisk
        public let simplificationSuggestions: [String]
        
        public init(
            originalGoal: String,
            subtasks: [Task],
            recommendedNext: UUID? = nil,
            estimatedTotal: TimeInterval,
            cognitiveLoadScore: Double,
            paralysisRisk: ParalysisRisk,
            simplificationSuggestions: [String] = []
        ) {
            self.originalGoal = originalGoal
            self.subtasks = subtasks
            self.recommendedNext = recommendedNext
            self.estimatedTotal = estimatedTotal
            self.cognitiveLoadScore = cognitiveLoadScore
            self.paralysisRisk = paralysisRisk
            self.simplificationSuggestions = simplificationSuggestions
        }
    }
    
    public enum ParalysisRisk: String, CaseIterable, Codable {
        case low        // Simple, clear path forward
        case moderate   // Some complexity but manageable
        case high       // Too many options/variables
        case critical   // User likely to abandon or make poor quick decision
    }
    
    // MARK: - Properties
    
    @Published public private(set) var currentBreakdown: BreakdownResult?
    @Published public private(set) var isProcessing = false
    
    private let cognitiveLoadThreshold: Double = 0.7
    private let maxSubtaskComplexity: TaskComplexity = .moderate
    
    // MARK: - Public Methods
    
    /// Main entry point: "I want to plan a trip from Nuremberg to Freising"
    public func breakdownGoal(_ goal: String) async -> BreakdownResult {
        isProcessing = true
        defer { isProcessing = false }
        
        // Analyze the goal for complexity patterns
        let analysis = analyzeGoalComplexity(goal)
        
        // Break it down based on the type
        let breakdown = await createBreakdown(for: goal, analysis: analysis)
        
        // Apply anti-paralysis optimizations
        let optimizedBreakdown = optimizeForActionability(breakdown)
        
        DispatchQueue.main.async {
            self.currentBreakdown = optimizedBreakdown
        }
        
        return optimizedBreakdown
    }
    
    /// Specifically for travel planning scenarios like your train example
    public func planTravel(from origin: String, to destination: String, when: Date? = nil) async -> BreakdownResult {
        let goal = "Plan travel from \(origin) to \(destination)"
        
        // Create travel-specific breakdown
        let subtasks: [Task] = [
            Task(
                title: "Get basic route options",
                description: "Find 2-3 main route options without detailed timing",
                complexity: .simple,
                estimatedTime: 300, // 5 minutes
                suggestedApproach: .delegate,
                automationLevel: .automatic
            ),
            Task(
                title: "Check Bahncard status",
                description: "Add Bahncard to DB app or verify discount",
                complexity: .simple,
                estimatedTime: 180, // 3 minutes
                suggestedApproach: .doNow,
                automationLevel: .guided
            ),
            Task(
                title: "Choose good-enough option",
                description: "Pick best route from options with reasonable price",
                complexity: .simple,
                estimatedTime: 300, // 5 minutes
                dependencies: [],
                suggestedApproach: .useDefault,
                automationLevel: .assisted
            ),
            Task(
                title: "Book ticket",
                description: "Complete booking with chosen option",
                complexity: .simple,
                estimatedTime: 300, // 5 minutes
                dependencies: [],
                suggestedApproach: .doNow,
                automationLevel: .manual
            )
        ]
        
        return BreakdownResult(
            originalGoal: goal,
            subtasks: subtasks,
            recommendedNext: subtasks.first?.id,
            estimatedTotal: subtasks.reduce(0) { $0 + $1.estimatedTime },
            cognitiveLoadScore: 0.3, // Low cognitive load
            paralysisRisk: .low,
            simplificationSuggestions: [
                "Don't optimize the route - any reasonable option is fine",
                "Set a 15-minute time limit for the entire process",
                "Use DB app defaults for most selections"
            ]
        )
    }
    
    // MARK: - Private Methods
    
    private func analyzeGoalComplexity(_ goal: String) -> GoalAnalysis {
        // Pattern recognition for common paralysis triggers
        let paralysisIndicators = [
            "plan", "optimize", "best", "perfect", "compare", "analyze",
            "research", "figure out", "decide between"
        ]
        
        let hasParalysisWords = paralysisIndicators.contains { goal.lowercased().contains($0) }
        let wordCount = goal.split(separator: " ").count
        
        return GoalAnalysis(
            hasParalysisIndicators: hasParalysisWords,
            estimatedComplexity: wordCount > 10 ? .complex : .moderate,
            domain: extractDomain(from: goal),
            urgency: .normal
        )
    }
    
    private func createBreakdown(for goal: String, analysis: GoalAnalysis) async -> BreakdownResult {
        // This would integrate with ML models to create intelligent breakdowns
        // For now, using pattern matching
        
        if analysis.domain == .travel {
            return await createTravelBreakdown(goal)
        } else if analysis.domain == .research {
            return await createResearchBreakdown(goal)
        } else {
            return await createGenericBreakdown(goal, analysis: analysis)
        }
    }
    
    private func createTravelBreakdown(_ goal: String) async -> BreakdownResult {
        // Extract origin/destination from goal
        // This is simplified - real implementation would use NLP
        return await planTravel(from: "Origin", to: "Destination")
    }
    
    private func createResearchBreakdown(_ goal: String) async -> BreakdownResult {
        let subtasks: [Task] = [
            Task(
                title: "Define specific question",
                description: "What exactly do you need to know?",
                complexity: .simple,
                estimatedTime: 300,
                suggestedApproach: .timeBox,
                automationLevel: .manual
            ),
            Task(
                title: "Set research time limit",
                description: "Decide max time to spend before making decision",
                complexity: .trivial,
                estimatedTime: 60,
                suggestedApproach: .doNow,
                automationLevel: .manual
            ),
            Task(
                title: "Gather information",
                description: "Collect relevant information within time limit",
                complexity: .moderate,
                estimatedTime: 900,
                dependencies: [],
                suggestedApproach: .delegate,
                automationLevel: .automatic
            )
        ]
        
        return BreakdownResult(
            originalGoal: goal,
            subtasks: subtasks,
            recommendedNext: subtasks.first?.id,
            estimatedTotal: subtasks.reduce(0) { $0 + $1.estimatedTime },
            cognitiveLoadScore: 0.4,
            paralysisRisk: .moderate,
            simplificationSuggestions: [
                "Set a timer before starting research",
                "Good enough information is usually sufficient",
                "Make decision with available information"
            ]
        )
    }
    
    private func createGenericBreakdown(_ goal: String, analysis: GoalAnalysis) async -> BreakdownResult {
        // Default breakdown for unknown domains
        let subtasks: [Task] = [
            Task(
                title: "Clarify the goal",
                description: "What is the minimum acceptable outcome?",
                complexity: .simple,
                estimatedTime: 300,
                suggestedApproach: .doNow,
                automationLevel: .manual
            ),
            Task(
                title: "Identify the first small step",
                description: "What's the smallest thing you can do right now?",
                complexity: .simple,
                estimatedTime: 300,
                suggestedApproach: .timeBox,
                automationLevel: .assisted
            )
        ]
        
        return BreakdownResult(
            originalGoal: goal,
            subtasks: subtasks,
            recommendedNext: subtasks.first?.id,
            estimatedTotal: 600,
            cognitiveLoadScore: 0.3,
            paralysisRisk: .low,
            simplificationSuggestions: [
                "Start with any reasonable first step",
                "Progress is more important than perfection"
            ]
        )
    }
    
    private func optimizeForActionability(_ breakdown: BreakdownResult) -> BreakdownResult {
        // Apply anti-paralysis heuristics
        
        var optimizedTasks = breakdown.subtasks
        
        // Ensure no task is too complex
        optimizedTasks = optimizedTasks.compactMap { task in
            if task.complexity == .overwhelming {
                // This task needs further breakdown
                return nil
            }
            return task
        }
        
        // Add time pressure for decision-heavy tasks
        optimizedTasks = optimizedTasks.map { task in
            if task.suggestedApproach == .simpleResearch {
                var modified = task
                modified = Task(
                    id: task.id,
                    title: task.title + " (5-min max)",
                    description: task.description + " - Set a timer!",
                    complexity: task.complexity,
                    estimatedTime: min(task.estimatedTime, 300),
                    dependencies: task.dependencies,
                    isBlocking: task.isBlocking,
                    suggestedApproach: .timeBox,
                    automationLevel: task.automationLevel
                )
                return modified
            }
            return task
        }
        
        return BreakdownResult(
            originalGoal: breakdown.originalGoal,
            subtasks: optimizedTasks,
            recommendedNext: breakdown.recommendedNext,
            estimatedTotal: optimizedTasks.reduce(0) { $0 + $1.estimatedTime },
            cognitiveLoadScore: max(0, breakdown.cognitiveLoadScore - 0.1), // Slight improvement
            paralysisRisk: breakdown.paralysisRisk,
            simplificationSuggestions: breakdown.simplificationSuggestions + [
                "Focus on progress, not perfection",
                "Any decision is better than no decision"
            ]
        )
    }
    
    private func extractDomain(from goal: String) -> GoalDomain {
        let travelWords = ["trip", "travel", "train", "flight", "book", "journey", "visit"]
        let researchWords = ["research", "find", "learn", "compare", "analyze", "study"]
        
        let lowercased = goal.lowercased()
        
        if travelWords.contains(where: { lowercased.contains($0) }) {
            return .travel
        } else if researchWords.contains(where: { lowercased.contains($0) }) {
            return .research
        } else {
            return .general
        }
    }
}

// MARK: - Supporting Types

private struct GoalAnalysis {
    let hasParalysisIndicators: Bool
    let estimatedComplexity: TaskBreakdownEngine.TaskComplexity
    let domain: GoalDomain
    let urgency: Urgency
}

private enum GoalDomain {
    case travel
    case research
    case planning
    case communication
    case finance
    case general
}

private enum Urgency {
    case low
    case normal
    case high
    case critical
} 