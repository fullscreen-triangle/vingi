import SwiftUI
import Combine

/// Example view demonstrating the complete cognitive pattern recognition system
/// Shows how all four patterns integrate in a real application
public struct CognitivePatternExampleView: View {
    @StateObject private var patternDetector = CognitivePatternDetector()
    @StateObject private var selectiveAbilityEngine = SelectiveAbilityRecognitionEngine()
    
    @State private var showingDashboard = false
    @State private var showingSelectiveAbilities = false
    @State private var isRunningExample = false
    
    public init() {}
    
    public var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    headerSection
                    
                    if isRunningExample {
                        runningExampleSection
                    } else {
                        exampleControlsSection
                    }
                    
                    detectedPatternsSection
                    
                    navigationLinksSection
                }
                .padding()
            }
            .navigationTitle("Cognitive Patterns Demo")
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title)
                    .foregroundColor(.blue)
                
                VStack(alignment: .leading) {
                    Text("Cognitive Pattern Recognition")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Text("Demonstrating all four cognitive inefficiency patterns")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Patterns Detected:")
                    .font(.headline)
                
                ForEach(["Analysis Paralysis", "Tunnel Vision", "Default Loops", "Selective Ability Self-Doubt"], id: \.self) { pattern in
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text(pattern)
                            .font(.body)
                    }
                }
            }
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var exampleControlsSection: some View {
        VStack(spacing: 16) {
            Text("Example Scenarios")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Run examples to see how each cognitive pattern is detected and managed")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 16) {
                ExampleScenarioCard(
                    title: "Analysis Paralysis",
                    description: "35 minutes researching restaurants, 12+ options considered",
                    icon: "clock.arrow.circlepath",
                    color: .orange
                ) {
                    runAnalysisParalysisExample()
                }
                
                ExampleScenarioCard(
                    title: "Tunnel Vision",
                    description: "2 hours on transportation, 3 minutes on food planning",
                    icon: "eye.trianglebadge.exclamationmark",
                    color: .red
                ) {
                    runTunnelVisionExample()
                }
                
                ExampleScenarioCard(
                    title: "Default Loops",
                    description: "Five Guys chosen 5 times in a row despite alternatives",
                    icon: "arrow.triangle.2.circlepath",
                    color: .purple
                ) {
                    runDefaultLoopExample()
                }
                
                ExampleScenarioCard(
                    title: "Ability Self-Doubt",
                    description: "80% accuracy with bank numbers, 30% confidence",
                    icon: "brain.head.profile",
                    color: .green
                ) {
                    runSelectiveAbilityExample()
                }
            }
            
            Button("Run All Examples") {
                runCompleteExample()
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(10)
        }
    }
    
    private var runningExampleSection: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.2)
            
            Text("Running cognitive pattern analysis...")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("This demonstrates real-time pattern detection and intervention suggestions")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var detectedPatternsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !patternDetector.detectedPatterns.isEmpty {
                Text("Currently Detected Patterns")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                ForEach(Array(patternDetector.detectedPatterns.enumerated()), id: \.offset) { index, pattern in
                    DetectedPatternCard(pattern: pattern)
                }
            }
            
            if let assessment = patternDetector.currentCognitiveLoad {
                CognitiveLoadCard(assessment: assessment)
            }
        }
    }
    
    private var navigationLinksSection: some View {
        VStack(spacing: 12) {
            Button("Open Full Dashboard") {
                showingDashboard = true
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(10)
            
            Button("Manage Selective Abilities") {
                showingSelectiveAbilities = true
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.green)
            .foregroundColor(.white)
            .cornerRadius(10)
        }
        .sheet(isPresented: $showingDashboard) {
            CognitivePatternDashboardView()
        }
        .sheet(isPresented: $showingSelectiveAbilities) {
            SelectiveAbilityView()
        }
    }
    
    // MARK: - Example Functions
    
    private func runAnalysisParalysisExample() {
        isRunningExample = true
        
        Task {
            // Simulate analysis paralysis scenario
            let decisions = [
                DecisionEvent(
                    domain: "Dining",
                    researchTime: 2100, // 35 minutes
                    optionsConsidered: 12,
                    decisionMade: false
                ),
                DecisionEvent(
                    domain: "Transportation",
                    researchTime: 1800, // 30 minutes
                    optionsConsidered: 15,
                    decisionMade: false
                )
            ]
            
            let behaviorData = UserBehaviorData(
                recentDecisions: decisions,
                planningActivity: [],
                routineChoices: [],
                performanceHistory: []
            )
            
            await patternDetector.detectActivePatterns(userBehavior: behaviorData)
            
            await MainActor.run {
                isRunningExample = false
            }
        }
    }
    
    private func runTunnelVisionExample() {
        isRunningExample = true
        
        Task {
            // Simulate tunnel vision scenario
            let planning = [
                PlanningEvent(
                    activity: "Transportation research",
                    domain: "Transportation",
                    timeSpent: 7200, // 2 hours
                    detailLevel: 0.9,
                    completeness: 0.85
                ),
                PlanningEvent(
                    activity: "Food planning",
                    domain: "Food",
                    timeSpent: 180, // 3 minutes
                    detailLevel: 0.1,
                    completeness: 0.1
                ),
                PlanningEvent(
                    activity: "Accommodation planning",
                    domain: "Accommodation",
                    timeSpent: 300, // 5 minutes
                    detailLevel: 0.2,
                    completeness: 0.15
                )
            ]
            
            let behaviorData = UserBehaviorData(
                recentDecisions: [],
                planningActivity: planning,
                routineChoices: [],
                performanceHistory: []
            )
            
            await patternDetector.detectActivePatterns(userBehavior: behaviorData)
            
            await MainActor.run {
                isRunningExample = false
            }
        }
    }
    
    private func runDefaultLoopExample() {
        isRunningExample = true
        
        Task {
            // Simulate default loop scenario
            let choices = [
                ChoiceEvent(domain: "Dining", choice: "Five Guys", alternatives: ["McDonald's", "Local restaurant"]),
                ChoiceEvent(domain: "Dining", choice: "Five Guys", alternatives: ["Subway", "Burger King"]),
                ChoiceEvent(domain: "Dining", choice: "Five Guys", alternatives: ["Local diner"]),
                ChoiceEvent(domain: "Dining", choice: "Five Guys", alternatives: ["McDonald's"]),
                ChoiceEvent(domain: "Dining", choice: "Five Guys", alternatives: ["Subway"])
            ]
            
            let behaviorData = UserBehaviorData(
                recentDecisions: [],
                planningActivity: [],
                routineChoices: choices,
                performanceHistory: []
            )
            
            await patternDetector.detectActivePatterns(userBehavior: behaviorData)
            
            await MainActor.run {
                isRunningExample = false
            }
        }
    }
    
    private func runSelectiveAbilityExample() {
        isRunningExample = true
        
        Task {
            // Simulate selective ability self-doubt scenario
            let performance = [
                SelectiveAbilityRecognitionEngine.PerformanceEvent(
                    domain: .meaningfulSequences,
                    context: .highStakes,
                    accuracy: 0.8, // 12 out of 15 digits recalled
                    confidence: 0.3, // Low confidence despite high performance
                    description: "Bank account number recall during phone verification"
                ),
                SelectiveAbilityRecognitionEngine.PerformanceEvent(
                    domain: .emotionalMemories,
                    context: .emotionallyCharged,
                    accuracy: 1.0, // Perfect recall
                    confidence: 0.95, // High confidence
                    description: "First words as a child: 'No, it was my idea'"
                ),
                SelectiveAbilityRecognitionEngine.PerformanceEvent(
                    domain: .routineInformation,
                    context: .routineTask,
                    accuracy: 0.2, // Poor password recall
                    confidence: 0.1, // Low confidence
                    description: "Password recall without backup"
                )
            ]
            
            let behaviorData = UserBehaviorData(
                recentDecisions: [],
                planningActivity: [],
                routineChoices: [],
                performanceHistory: performance
            )
            
            await patternDetector.detectActivePatterns(userBehavior: behaviorData)
            
            // Also update the selective ability engine
            for event in performance {
                selectiveAbilityEngine.recordPerformanceEvent(event)
            }
            
            await MainActor.run {
                isRunningExample = false
            }
        }
    }
    
    private func runCompleteExample() {
        isRunningExample = true
        
        Task {
            // Run all examples in sequence
            await runAnalysisParalysisExample()
            await runTunnelVisionExample()
            await runDefaultLoopExample()
            await runSelectiveAbilityExample()
            
            await MainActor.run {
                isRunningExample = false
            }
        }
    }
}

// MARK: - Supporting Views

struct ExampleScenarioCard: View {
    let title: String
    let description: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: icon)
                        .font(.title2)
                        .foregroundColor(color)
                    
                    Spacer()
                }
                
                Text(title)
                    .font(.headline)
                    .foregroundColor(.primary)
                    .multilineTextAlignment(.leading)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(3)
                    .multilineTextAlignment(.leading)
                
                Spacer()
            }
            .padding()
            .frame(height: 140)
            .background(color.opacity(0.1))
            .cornerRadius(12)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct DetectedPatternCard: View {
    let pattern: CognitivePattern
    
    var body: some View {
        HStack {
            Image(systemName: patternIcon)
                .font(.title2)
                .foregroundColor(patternColor)
                .frame(width: 40)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(patternTitle)
                    .font(.headline)
                
                Text(patternDescription)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                
                if !interventions.isEmpty {
                    Text("Suggestion: \(interventions.first ?? "")")
                        .font(.caption)
                        .foregroundColor(.blue)
                        .lineLimit(1)
                }
            }
            
            Spacer()
        }
        .padding()
        .background(patternColor.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var patternTitle: String {
        switch pattern {
        case .analysisParalysis: return "Analysis Paralysis Detected"
        case .tunnelVision: return "Tunnel Vision Detected"
        case .defaultLoop: return "Default Loop Detected"
        case .exceptionalAbilitySelfDoubt: return "Ability Self-Doubt Detected"
        }
    }
    
    private var patternDescription: String {
        switch pattern {
        case .analysisParalysis(_, let domain, _):
            return "Excessive optimization in \(domain)"
        case .tunnelVision(let focus, let neglected, _):
            return "Over-focusing on \(focus), neglecting \(neglected.joined(separator: ", "))"
        case .defaultLoop(let domain, let constraint, let potential):
            return "\(domain): \(constraint.rawValue) constraint, \(Int(potential * 100))% optimization potential"
        case .exceptionalAbilitySelfDoubt(let ability, _, let actual, let perceived, _):
            return "\(ability.description): \(Int(actual * 100))% actual vs \(Int(perceived * 100))% perceived"
        }
    }
    
    private var patternIcon: String {
        switch pattern {
        case .analysisParalysis: return "clock.arrow.circlepath"
        case .tunnelVision: return "eye.trianglebadge.exclamationmark"
        case .defaultLoop: return "arrow.triangle.2.circlepath"
        case .exceptionalAbilitySelfDoubt: return "brain.head.profile"
        }
    }
    
    private var patternColor: Color {
        switch pattern {
        case .analysisParalysis: return .orange
        case .tunnelVision: return .red
        case .defaultLoop: return .purple
        case .exceptionalAbilitySelfDoubt: return .green
        }
    }
    
    private var interventions: [String] {
        switch pattern {
        case .analysisParalysis(_, _, let interventions):
            return interventions
        case .tunnelVision:
            return ["Review all planning domains", "Set time limits per domain"]
        case .defaultLoop:
            return ["Explore safe alternatives", "Try new options gradually"]
        case .exceptionalAbilitySelfDoubt:
            return ["Trust your validated abilities", "Document successful outcomes"]
        }
    }
}

struct CognitiveLoadCard: View {
    let assessment: CognitiveLoadAssessment
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Current Cognitive Load")
                .font(.headline)
            
            HStack {
                Text("Overall Load: \(Int(assessment.overallLoad * 100))%")
                    .font(.body)
                
                Spacer()
                
                Text(assessment.urgencyLevel.rawValue.capitalized)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(urgencyColor.opacity(0.2))
                    .foregroundColor(urgencyColor)
                    .cornerRadius(8)
            }
            
            if !assessment.recommendedInterventions.isEmpty {
                Text("Recommendations:")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                ForEach(assessment.recommendedInterventions.prefix(3), id: \.self) { intervention in
                    Text("• \(intervention)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var urgencyColor: Color {
        switch assessment.urgencyLevel {
        case .monitoring: return .green
        case .attention: return .orange
        case .intervention: return .red
        case .emergency: return .purple
        }
    }
}

#Preview {
    CognitivePatternExampleView()
} 