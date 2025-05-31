import SwiftUI
import Combine

/// Comprehensive dashboard for monitoring and managing all four cognitive patterns
public struct CognitivePatternDashboardView: View {
    @StateObject private var patternDetector = CognitivePatternDetector()
    @StateObject private var selectiveAbilityEngine = SelectiveAbilityRecognitionEngine()
    @State private var selectedPattern: CognitivePattern?
    @State private var showingDetailView = false
    
    public init() {}
    
    public var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    headerSection
                    
                    if patternDetector.isAnalyzing {
                        analysisProgressView
                    } else {
                        mainContent
                    }
                }
                .padding()
            }
            .navigationTitle("Cognitive Patterns")
            .refreshable {
                await refreshPatternAnalysis()
            }
        }
        .sheet(isPresented: $showingDetailView) {
            if let pattern = selectedPattern {
                PatternDetailView(pattern: pattern)
            }
        }
        .onAppear {
            Task {
                await refreshPatternAnalysis()
            }
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundColor(.blue)
                
                VStack(alignment: .leading) {
                    Text("Cognitive Pattern Monitor")
                        .font(.headline)
                    Text("Real-time detection of cognitive inefficiency patterns")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            
            if let assessment = patternDetector.currentCognitiveLoad {
                CognitiveLoadSummaryView(assessment: assessment)
            }
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }
    
    @ViewBuilder
    private var mainContent: some View {
        // Active Patterns Section
        if !patternDetector.detectedPatterns.isEmpty {
            activePatternSection
        }
        
        // Pattern Risk Assessment
        patternRiskSection
        
        // Individual Pattern Modules
        patternModulesSection
        
        // Quick Actions
        quickActionsSection
    }
    
    private var analysisProgressView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.2)
            
            Text("Analyzing cognitive patterns...")
                .font(.headline)
                .foregroundColor(.secondary)
        }
        .frame(height: 200)
    }
    
    private var activePatternSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Active Patterns")
                .font(.title2)
                .fontWeight(.semibold)
            
            ForEach(Array(patternDetector.detectedPatterns.enumerated()), id: \.offset) { index, pattern in
                ActivePatternCard(
                    pattern: pattern,
                    onTap: {
                        selectedPattern = pattern
                        showingDetailView = true
                    }
                )
            }
        }
    }
    
    private var patternRiskSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Pattern Risk Assessment")
                .font(.title2)
                .fontWeight(.semibold)
            
            if let assessment = patternDetector.currentCognitiveLoad {
                PatternRiskGrid(assessment: assessment)
            }
        }
    }
    
    private var patternModulesSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Pattern Management")
                .font(.title2)
                .fontWeight(.semibold)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 16) {
                PatternModuleCard(
                    title: "Analysis Paralysis",
                    icon: "clock.arrow.circlepath",
                    color: .orange,
                    description: "Decision optimization & time boxing"
                )
                
                PatternModuleCard(
                    title: "Tunnel Vision",
                    icon: "eye.trianglebadge.exclamationmark",
                    color: .red,
                    description: "Comprehensive planning coverage"
                )
                
                PatternModuleCard(
                    title: "Default Loops",
                    icon: "arrow.triangle.2.circlepath",
                    color: .purple,
                    description: "Safe exploration & optimization"
                )
                
                PatternModuleCard(
                    title: "Selective Abilities",
                    icon: "brain.head.profile",
                    color: .green,
                    description: "Confidence building & validation"
                ) {
                    // Navigate to SelectiveAbilityView
                }
            }
        }
    }
    
    private var quickActionsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Quick Actions")
                .font(.title2)
                .fontWeight(.semibold)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                QuickActionButton(
                    title: "Run Full Scan",
                    icon: "brain.head.profile",
                    action: {
                        Task { await refreshPatternAnalysis() }
                    }
                )
                
                QuickActionButton(
                    title: "Reset Patterns",
                    icon: "arrow.clockwise",
                    action: {
                        // Reset pattern detection
                    }
                )
                
                QuickActionButton(
                    title: "Export Data",
                    icon: "square.and.arrow.up",
                    action: {
                        // Export pattern data
                    }
                )
                
                QuickActionButton(
                    title: "Settings",
                    icon: "gearshape",
                    action: {
                        // Open settings
                    }
                )
            }
        }
    }
    
    private func refreshPatternAnalysis() async {
        // Create sample user behavior data for demonstration
        let sampleBehavior = UserBehaviorData(
            recentDecisions: createSampleDecisions(),
            planningActivity: createSamplePlanning(),
            routineChoices: createSampleChoices(),
            performanceHistory: createSamplePerformance()
        )
        
        let userContext = UserContext(
            currentTask: "Planning weekend trip",
            recentDecisionCount: 5,
            contextSwitchCount: 3,
            researchTime: 1800, // 30 minutes
            planningActivity: sampleBehavior.planningActivity,
            routineChoices: sampleBehavior.routineChoices,
            performanceHistory: sampleBehavior.performanceHistory
        )
        
        // Run pattern detection
        await patternDetector.detectActivePatterns(userBehavior: sampleBehavior)
        await patternDetector.assessCognitiveLoad(userContext: userContext)
    }
    
    // Sample data creation methods
    private func createSampleDecisions() -> [DecisionEvent] {
        return [
            DecisionEvent(
                domain: "Transportation",
                researchTime: 1800, // 30 minutes
                optionsConsidered: 12,
                decisionMade: false
            ),
            DecisionEvent(
                domain: "Accommodation",
                researchTime: 2700, // 45 minutes
                optionsConsidered: 8,
                decisionMade: true,
                satisfaction: 0.7
            )
        ]
    }
    
    private func createSamplePlanning() -> [PlanningEvent] {
        return [
            PlanningEvent(
                activity: "Transportation research",
                domain: "Transportation",
                timeSpent: 3600, // 1 hour
                detailLevel: 0.9,
                completeness: 0.8
            ),
            PlanningEvent(
                activity: "Activity planning",
                domain: "Activities",
                timeSpent: 1800, // 30 minutes
                detailLevel: 0.9,
                completeness: 0.7
            ),
            PlanningEvent(
                activity: "Food planning",
                domain: "Food",
                timeSpent: 300, // 5 minutes
                detailLevel: 0.2,
                completeness: 0.1
            )
        ]
    }
    
    private func createSampleChoices() -> [ChoiceEvent] {
        return [
            ChoiceEvent(
                domain: "Dining",
                choice: "Five Guys",
                alternatives: ["McDonald's", "Burger King", "Local restaurant"],
                satisfaction: 0.7,
                metadata: ["location": "Nuremberg City Center"]
            ),
            ChoiceEvent(
                domain: "Dining",
                choice: "Five Guys",
                alternatives: ["McDonald's", "Subway"],
                satisfaction: 0.6,
                metadata: ["location": "Nuremberg City Center"]
            ),
            ChoiceEvent(
                domain: "Shopping",
                choice: "1-minute supermarket",
                alternatives: ["Larger supermarket", "Specialty shops"],
                satisfaction: 0.5,
                metadata: ["convenience": "1-minute walk"]
            )
        ]
    }
    
    private func createSamplePerformance() -> [SelectiveAbilityRecognitionEngine.PerformanceEvent] {
        return [
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .meaningfulSequences,
                context: .highStakes,
                accuracy: 0.8,
                confidence: 0.3,
                description: "Bank number recall during phone call"
            ),
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .emotionalMemories,
                context: .emotionallyCharged,
                accuracy: 1.0,
                confidence: 0.95,
                description: "First words as a child"
            ),
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .routineInformation,
                context: .routineTask,
                accuracy: 0.2,
                confidence: 0.1,
                description: "Password recall"
            )
        ]
    }
}

// MARK: - Supporting Views

struct CognitiveLoadSummaryView: View {
    let assessment: CognitiveLoadAssessment
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Cognitive Load")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text("\(Int(assessment.overallLoad * 100))%")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(loadColor)
            }
            
            Spacer()
            
            VStack(alignment: .trailing) {
                Text(assessment.urgencyLevel.rawValue.capitalized)
                    .font(.caption)
                    .foregroundColor(urgencyColor)
                
                Circle()
                    .fill(urgencyColor)
                    .frame(width: 12, height: 12)
            }
        }
    }
    
    private var loadColor: Color {
        switch assessment.overallLoad {
        case 0..<0.3: return .green
        case 0.3..<0.6: return .orange
        default: return .red
        }
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

struct ActivePatternCard: View {
    let pattern: CognitivePattern
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack {
                Image(systemName: patternIcon)
                    .font(.title2)
                    .foregroundColor(patternColor)
                    .frame(width: 40)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(patternTitle)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text(patternDescription)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
                
                Spacer()
                
                Image(systemName: "chevron.right")
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(patternColor.opacity(0.1))
            .cornerRadius(12)
        }
        .buttonStyle(PlainButtonStyle())
    }
    
    private var patternTitle: String {
        switch pattern {
        case .analysisParalysis: return "Analysis Paralysis"
        case .tunnelVision: return "Tunnel Vision"
        case .defaultLoop: return "Default Loop"
        case .exceptionalAbilitySelfDoubt: return "Ability Self-Doubt"
        }
    }
    
    private var patternDescription: String {
        switch pattern {
        case .analysisParalysis(_, let domain, _):
            return "Excessive optimization in \(domain)"
        case .tunnelVision(let focus, _, _):
            return "Over-focusing on \(focus)"
        case .defaultLoop(let domain, _, _):
            return "Repetitive choices in \(domain)"
        case .exceptionalAbilitySelfDoubt(let ability, _, _, _, _):
            return "Doubting proven ability: \(ability.description)"
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
}

struct PatternRiskGrid: View {
    let assessment: CognitiveLoadAssessment
    
    var body: some View {
        LazyVGrid(columns: [
            GridItem(.flexible()),
            GridItem(.flexible())
        ], spacing: 12) {
            ForEach(Array(assessment.patternRisks.keys.sorted()), id: \.self) { patternType in
                if let risk = assessment.patternRisks[patternType] {
                    PatternRiskCard(
                        patternType: patternType,
                        risk: risk
                    )
                }
            }
        }
    }
}

struct PatternRiskCard: View {
    let patternType: String
    let risk: Double
    
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text(formattedPatternName)
                    .font(.caption)
                    .fontWeight(.medium)
                    .lineLimit(1)
                
                Spacer()
                
                Text("\(Int(risk * 100))%")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(riskColor)
            }
            
            ProgressView(value: risk)
                .progressViewStyle(LinearProgressViewStyle(tint: riskColor))
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
    
    private var formattedPatternName: String {
        patternType.replacingOccurrences(of: "_", with: " ").capitalized
    }
    
    private var riskColor: Color {
        switch risk {
        case 0..<0.3: return .green
        case 0.3..<0.6: return .orange
        default: return .red
        }
    }
}

struct PatternModuleCard: View {
    let title: String
    let icon: String
    let color: Color
    let description: String
    var action: (() -> Void)? = nil
    
    var body: some View {
        Button(action: action ?? {}) {
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
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                
                Spacer()
            }
            .padding()
            .frame(height: 120)
            .background(color.opacity(0.1))
            .cornerRadius(12)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct QuickActionButton: View {
    let title: String
    let icon: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: icon)
                    .font(.caption)
                
                Text(title)
                    .font(.caption)
                    .fontWeight(.medium)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(Color.blue.opacity(0.1))
            .foregroundColor(.blue)
            .cornerRadius(8)
        }
    }
}

struct PatternDetailView: View {
    let pattern: CognitivePattern
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Pattern details based on type
                    switch pattern {
                    case .analysisParalysis(let severity, let domain, let interventions):
                        AnalysisParalysisDetailView(
                            severity: severity,
                            domain: domain,
                            interventions: interventions
                        )
                        
                    case .tunnelVision(let focusDomain, let neglectedDomains, let riskLevel):
                        TunnelVisionDetailView(
                            focusDomain: focusDomain,
                            neglectedDomains: neglectedDomains,
                            riskLevel: riskLevel
                        )
                        
                    case .defaultLoop(let domain, let constraintType, let optimizationPotential):
                        DefaultLoopDetailView(
                            domain: domain,
                            constraintType: constraintType,
                            optimizationPotential: optimizationPotential
                        )
                        
                    case .exceptionalAbilitySelfDoubt(let ability, let evidenceCount, let actualPerformance, let perceivedPerformance, let confidenceGap):
                        ExceptionalAbilityDetailView(
                            ability: ability,
                            evidenceCount: evidenceCount,
                            actualPerformance: actualPerformance,
                            perceivedPerformance: perceivedPerformance,
                            confidenceGap: confidenceGap
                        )
                    }
                }
                .padding()
            }
            .navigationTitle("Pattern Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// Pattern-specific detail views would be implemented here...
struct AnalysisParalysisDetailView: View {
    let severity: ParalysisSeverity
    let domain: String
    let interventions: [String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Analysis Paralysis Detected")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Severity: \(severity.rawValue.capitalized)")
            Text("Domain: \(domain)")
            
            Text("Recommended Interventions:")
                .font(.headline)
            
            ForEach(interventions, id: \.self) { intervention in
                Text("• \(intervention)")
            }
        }
    }
}

struct TunnelVisionDetailView: View {
    let focusDomain: String
    let neglectedDomains: [String]
    let riskLevel: TunnelRisk
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Tunnel Vision Detected")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Focus Domain: \(focusDomain)")
            Text("Risk Level: \(riskLevel.rawValue.capitalized)")
            
            Text("Neglected Domains:")
                .font(.headline)
            
            ForEach(neglectedDomains, id: \.self) { domain in
                Text("• \(domain)")
            }
        }
    }
}

struct DefaultLoopDetailView: View {
    let domain: String
    let constraintType: ConstraintType
    let optimizationPotential: Double
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Default Loop Detected")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Domain: \(domain)")
            Text("Constraint: \(constraintType.rawValue.capitalized)")
            Text("Optimization Potential: \(Int(optimizationPotential * 100))%")
        }
    }
}

struct ExceptionalAbilityDetailView: View {
    let ability: SelectiveAbilityRecognitionEngine.AbilityDomain
    let evidenceCount: Int
    let actualPerformance: Double
    let perceivedPerformance: Double
    let confidenceGap: Double
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Exceptional Ability Self-Doubt")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Ability: \(ability.description)")
            Text("Evidence Count: \(evidenceCount)")
            Text("Actual Performance: \(Int(actualPerformance * 100))%")
            Text("Perceived Performance: \(Int(perceivedPerformance * 100))%")
            Text("Confidence Gap: \(Int(confidenceGap * 100))%")
        }
    }
}

#Preview {
    CognitivePatternDashboardView()
} 