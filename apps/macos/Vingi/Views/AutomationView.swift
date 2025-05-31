import SwiftUI
import VingiCore

struct AutomationView: View {
    @StateObject private var taskBreakdown = TaskBreakdownEngine()
    @StateObject private var decisionEngine = DecisionEngine()
    
    @State private var currentGoal: String = "Plan trip to Ansbach with friends"
    @State private var showingBreakdown = false
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                headerSection
                
                if let breakdown = taskBreakdown.currentBreakdown {
                    breakdownResultsSection(breakdown)
                } else {
                    exampleScenarioSection
                }
                
                demoScenariosSection
                Spacer()
            }
            .padding()
        }
        .navigationTitle("Task Automation")
        .onAppear {
            loadCurrentTask()
        }
    }
    
    // MARK: - Header Section
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Prevent Analysis Paralysis")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Turn overwhelming goals into simple, actionable steps")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            HStack {
                TextField("What do you want to accomplish?", text: $currentGoal)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                
                Button("Break It Down") {
                    Task {
                        await breakdownGoal()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(taskBreakdown.isProcessing)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    // MARK: - Example Scenario
    
    private var exampleScenarioSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Real Example: The Ansbach Trip Disaster")
                .font(.headline)
                .foregroundColor(.red)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("❌ What Happened:")
                    .fontWeight(.semibold)
                
                Text("• Planned transportation and sightseeing perfectly")
                Text("• Completely forgot about food planning")
                Text("• Arrived hungry, all restaurants closed")
                Text("• Entire trip ruined by one oversight")
            }
            .padding()
            .background(Color.red.opacity(0.1))
            .cornerRadius(8)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("✅ How Vingi Would Have Helped:")
                    .fontWeight(.semibold)
                
                Text("• Automatic comprehensive planning checklist")
                Text("• 'Practical necessities' always included")
                Text("• Restaurant hours checked in advance")
                Text("• Backup food options identified")
            }
            .padding()
            .background(Color.green.opacity(0.1))
            .cornerRadius(8)
            
            Button("See How Vingi Breaks Down 'Plan trip to Ansbach'") {
                currentGoal = "Plan trip to Ansbach with friends to see walled city"
                Task {
                    await breakdownGoal()
                }
            }
            .buttonStyle(.bordered)
        }
    }
    
    // MARK: - Breakdown Results
    
    private func breakdownResultsSection(_ breakdown: TaskBreakdownEngine.BreakdownResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            // Risk Assessment
            riskAssessmentCard(breakdown)
            
            // Simplification Suggestions
            if !breakdown.simplificationSuggestions.isEmpty {
                simplificationCard(breakdown.simplificationSuggestions)
            }
            
            // Task Breakdown
            taskBreakdownCard(breakdown)
            
            // Cognitive Load Meter
            cognitiveLoadMeter(breakdown.cognitiveLoadScore)
        }
    }
    
    private func riskAssessmentCard(_ breakdown: TaskBreakdownEngine.BreakdownResult) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: riskIcon(breakdown.paralysisRisk))
                    .foregroundColor(riskColor(breakdown.paralysisRisk))
                Text("Analysis Paralysis Risk: \(breakdown.paralysisRisk.rawValue.capitalized)")
                    .fontWeight(.semibold)
            }
            
            Text("Total estimated time: \(formatTime(breakdown.estimatedTotal))")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(riskColor(breakdown.paralysisRisk).opacity(0.1))
        .cornerRadius(8)
    }
    
    private func simplificationCard(_ suggestions: [String]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "lightbulb")
                    .foregroundColor(.orange)
                Text("Anti-Paralysis Tips")
                    .fontWeight(.semibold)
            }
            
            ForEach(suggestions, id: \.self) { suggestion in
                HStack(alignment: .top) {
                    Text("•")
                    Text(suggestion)
                        .font(.subheadline)
                }
            }
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(8)
    }
    
    private func taskBreakdownCard(_ breakdown: TaskBreakdownEngine.BreakdownResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Actionable Steps")
                .font(.headline)
            
            ForEach(breakdown.subtasks) { task in
                taskRow(task, isRecommended: task.id == breakdown.recommendedNext)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func taskRow(_ task: TaskBreakdownEngine.Task, isRecommended: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Circle()
                    .fill(isRecommended ? Color.blue : Color.gray)
                    .frame(width: 8, height: 8)
                
                Text(task.title)
                    .fontWeight(isRecommended ? .semibold : .regular)
                
                Spacer()
                
                automationBadge(task.automationLevel)
                approachBadge(task.suggestedApproach)
            }
            
            Text(task.description)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .padding(.leading, 16)
            
            HStack {
                Text("⏱ \(formatTime(task.estimatedTime))")
                Text("🧠 \(task.complexity.rawValue)")
                
                if isRecommended {
                    Text("👆 Start here")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color.blue.opacity(0.2))
                        .cornerRadius(4)
                }
            }
            .font(.caption)
            .foregroundColor(.secondary)
            .padding(.leading, 16)
        }
        .padding()
        .background(isRecommended ? Color.blue.opacity(0.05) : Color.clear)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isRecommended ? Color.blue : Color.clear, lineWidth: 1)
        )
    }
    
    private func automationBadge(_ level: TaskBreakdownEngine.AutomationLevel) -> some View {
        Text(automationText(level))
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(automationColor(level))
            .foregroundColor(.white)
            .cornerRadius(4)
    }
    
    private func approachBadge(_ approach: TaskBreakdownEngine.TaskApproach) -> some View {
        Text(approachText(approach))
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(approachColor(approach))
            .foregroundColor(.white)
            .cornerRadius(4)
    }
    
    private func cognitiveLoadMeter(_ score: Double) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Cognitive Load")
                    .fontWeight(.semibold)
                Spacer()
                Text("\(Int(score * 100))%")
                    .fontWeight(.semibold)
                    .foregroundColor(loadColor(score))
            }
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color(.systemGray5))
                        .frame(height: 8)
                    
                    Rectangle()
                        .fill(loadColor(score))
                        .frame(width: geometry.size.width * score, height: 8)
                }
            }
            .frame(height: 8)
            .cornerRadius(4)
            
            Text(loadDescription(score))
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
    
    // MARK: - Demo Scenarios
    
    private var demoScenariosSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("How Vingi Prevents Decision Disasters")
                .font(.headline)
            
            ansaschTripDemo
            
            shoppingOptimizationDemo
        }
    }
    
    private var shoppingOptimizationDemo: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("🛒 Shopping Constraint Problem")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.orange)
            
            Text("\"I only go to the 1-minute supermarket because I've optimized my preferences to what's on their shelves. I want better products but don't want complexity.\"")
                .font(.body)
                .italic()
            
            Text("Vingi's Multi-Stop Solution:")
                .fontWeight(.semibold)
                .foregroundColor(.green)
            
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top) {
                    Text("•")
                    Text("Monday: Fresh milk from dairy shop (3 min walk) - better quality, see neighborhood")
                        .font(.subheadline)
                }
                
                HStack(alignment: .top) {
                    Text("•")
                    Text("Wednesday: Real bread from bakery (8 min walk) - discover coffee place next door")
                        .font(.subheadline)
                }
                
                HStack(alignment: .top) {
                    Text("•")
                    Text("Saturday: Main groceries at larger store (15 min walk) - better selection, same brands")
                        .font(.subheadline)
                }
                
                HStack(alignment: .top) {
                    Text("•")
                    Text("Emergency backup: 1-minute supermarket always there when needed")
                        .font(.subheadline)
                        .foregroundColor(.blue)
                }
            }
            
            Text("Result: Same budget, same time commitment, but better products + 15-20 minutes more walking + discovery opportunities")
                .font(.subheadline)
                .foregroundColor(.green)
                .fontWeight(.semibold)
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var ansaschTripDemo: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("🍽 The Ansbach Trip Disaster")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.red)
            
            Text("\"Planned perfect transportation and sightseeing for Ansbach, but forgot about food. Arrived hungry, everything was closed, ruined the whole trip.\"")
                .font(.body)
                .italic()
            
            Text("How Vingi Would Have Prevented This:")
                .fontWeight(.semibold)
                .foregroundColor(.green)
            
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top) {
                    Text("1.")
                    Text("Food planning comes FIRST, not last")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
                
                HStack(alignment: .top) {
                    Text("2.")
                    Text("Check restaurant hours for your specific visit day")
                        .font(.subheadline)
                }
                
                HStack(alignment: .top) {
                    Text("3.")
                    Text("Identify backup options: grocery store, food trucks, cafes")
                        .font(.subheadline)
                }
                
                HStack(alignment: .top) {
                    Text("4.")
                    Text("Pack emergency snacks as insurance")
                        .font(.subheadline)
                }
            }
            
            Text("Result: Great trip with satisfied hunger, transportation becomes secondary concern")
                .font(.subheadline)
                .foregroundColor(.green)
                .fontWeight(.semibold)
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(12)
    }
    
    // MARK: - Helper Methods
    
    private func breakdownGoal() async {
        // Simulate the Ansbach trip breakdown
        if currentGoal.lowercased().contains("ansbach") {
            let breakdown = await createAnsbachTripBreakdown()
            DispatchQueue.main.async {
                self.taskBreakdown.currentBreakdown = breakdown
            }
        } else {
            await taskBreakdown.breakdownGoal(currentGoal)
        }
    }
    
    private func createAnsbachTripBreakdown() async -> TaskBreakdownEngine.BreakdownResult {
        let subtasks: [TaskBreakdownEngine.Task] = [
            TaskBreakdownEngine.Task(
                title: "Check restaurant hours & options",
                description: "Research 3-4 restaurants in Ansbach, check opening hours for your visit day, identify backup options",
                complexity: .simple,
                estimatedTime: 600, // 10 minutes
                suggestedApproach: .delegate,
                automationLevel: .automatic
            ),
            TaskBreakdownEngine.Task(
                title: "Plan transportation",
                description: "Book train tickets or plan driving route with parking info",
                complexity: .simple,
                estimatedTime: 900, // 15 minutes
                suggestedApproach: .useDefault,
                automationLevel: .assisted
            ),
            TaskBreakdownEngine.Task(
                title: "Create backup food plan",
                description: "Identify grocery stores, food trucks, or cafes as Plan B. Pack snacks just in case.",
                complexity: .trivial,
                estimatedTime: 300, // 5 minutes
                suggestedApproach: .doNow,
                automationLevel: .manual
            ),
            TaskBreakdownEngine.Task(
                title: "Check walled city opening hours",
                description: "Verify attraction hours and any special events or closures",
                complexity: .trivial,
                estimatedTime: 180, // 3 minutes
                suggestedApproach: .delegate,
                automationLevel: .automatic
            ),
            TaskBreakdownEngine.Task(
                title: "Coordinate with friends",
                description: "Share itinerary, confirm meeting point and times",
                complexity: .simple,
                estimatedTime: 300, // 5 minutes
                suggestedApproach: .doNow,
                automationLevel: .manual
            )
        ]
        
        return TaskBreakdownEngine.BreakdownResult(
            originalGoal: currentGoal,
            subtasks: subtasks,
            recommendedNext: subtasks.first?.id,
            estimatedTotal: subtasks.reduce(0) { $0 + $1.estimatedTime },
            cognitiveLoadScore: 0.25, // Very low cognitive load
            paralysisRisk: .low,
            simplificationSuggestions: [
                "Food planning comes FIRST - everything else is secondary",
                "Always have a backup food option when visiting small towns",
                "10 minutes of restaurant research prevents hours of hunger",
                "Pack snacks as insurance against closed restaurants"
            ]
        )
    }
    
    // MARK: - UI Helper Functions
    
    private func riskIcon(_ risk: TaskBreakdownEngine.ParalysisRisk) -> String {
        switch risk {
        case .low: return "checkmark.circle"
        case .moderate: return "exclamationmark.triangle"
        case .high: return "xmark.circle"
        case .critical: return "flame"
        }
    }
    
    private func riskColor(_ risk: TaskBreakdownEngine.ParalysisRisk) -> Color {
        switch risk {
        case .low: return .green
        case .moderate: return .orange
        case .high: return .red
        case .critical: return .purple
        }
    }
    
    private func automationText(_ level: TaskBreakdownEngine.AutomationLevel) -> String {
        switch level {
        case .manual: return "Manual"
        case .assisted: return "Assisted"
        case .guided: return "Guided"
        case .automatic: return "Auto"
        }
    }
    
    private func automationColor(_ level: TaskBreakdownEngine.AutomationLevel) -> Color {
        switch level {
        case .manual: return .gray
        case .assisted: return .blue
        case .guided: return .orange
        case .automatic: return .green
        }
    }
    
    private func approachText(_ approach: TaskBreakdownEngine.TaskApproach) -> String {
        switch approach {
        case .doNow: return "Do Now"
        case .useDefault: return "Use Default"
        case .simpleResearch: return "Quick Research"
        case .delegate: return "Let Vingi Handle"
        case .timeBox: return "Time Box"
        case .goodEnough: return "Good Enough"
        }
    }
    
    private func approachColor(_ approach: TaskBreakdownEngine.TaskApproach) -> Color {
        switch approach {
        case .doNow: return .red
        case .useDefault: return .blue
        case .simpleResearch: return .orange
        case .delegate: return .green
        case .timeBox: return .purple
        case .goodEnough: return .brown
        }
    }
    
    private func loadColor(_ score: Double) -> Color {
        if score < 0.3 { return .green }
        else if score < 0.6 { return .yellow }
        else if score < 0.8 { return .orange }
        else { return .red }
    }
    
    private func loadDescription(_ score: Double) -> String {
        if score < 0.3 { return "Low cognitive load - easy to execute" }
        else if score < 0.6 { return "Moderate load - manageable with focus" }
        else if score < 0.8 { return "High load - risk of analysis paralysis" }
        else { return "Overwhelming - needs further breakdown" }
    }
    
    private func formatTime(_ seconds: TimeInterval) -> String {
        let minutes = Int(seconds / 60)
        if minutes < 60 {
            return "\(minutes)m"
        } else {
            let hours = minutes / 60
            let remainingMinutes = minutes % 60
            return "\(hours)h \(remainingMinutes)m"
        }
    }
}

#Preview {
    AutomationView()
}
