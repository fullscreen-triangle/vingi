import SwiftUI
import Combine

/// SwiftUI view for the Selective Ability Recognition Engine
/// Helps users understand and trust their selective cognitive strengths
public struct SelectiveAbilityView: View {
    @StateObject private var engine = SelectiveAbilityRecognitionEngine()
    @State private var selectedDomain: SelectiveAbilityRecognitionEngine.AbilityDomain?
    @State private var selectedContext: SelectiveAbilityRecognitionEngine.ContextType?
    @State private var showingAddEvent = false
    @State private var showingConfidenceStrategy = false
    @State private var currentStrategy: SelectiveAbilityRecognitionEngine.ConfidenceStrategy?
    
    public init() {}
    
    public var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    headerSection
                    
                    if engine.isAnalyzing {
                        analysisProgressView
                    } else {
                        mainContent
                    }
                }
                .padding()
            }
            .navigationTitle("Selective Abilities")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: { showingAddEvent = true }) {
                        Image(systemName: "plus.circle.fill")
                    }
                }
            }
        }
        .sheet(isPresented: $showingAddEvent) {
            AddPerformanceEventView(engine: engine)
        }
        .sheet(isPresented: $showingConfidenceStrategy) {
            if let strategy = currentStrategy {
                ConfidenceStrategyView(strategy: strategy)
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
                    Text("Selective Cognitive Abilities")
                        .font(.headline)
                    Text("Recognize and trust your domain-specific strengths")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            
            if let abilityMap = engine.currentAbilityMap {
                OverallConfidenceView(abilityMap: abilityMap)
            }
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }
    
    @ViewBuilder
    private var mainContent: some View {
        if let abilityMap = engine.currentAbilityMap {
            domainGrid(abilityMap: abilityMap)
            
            if !engine.recentValidations.isEmpty {
                recentValidationsSection
            }
            
            confidenceBuilderSection
        } else {
            emptyStateView
        }
    }
    
    private var analysisProgressView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.2)
            
            Text("Analyzing ability patterns...")
                .font(.headline)
                .foregroundColor(.secondary)
        }
        .frame(height: 200)
    }
    
    private func domainGrid(abilityMap: SelectiveAbilityRecognitionEngine.AbilityMap) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Ability Domains")
                .font(.title2)
                .fontWeight(.semibold)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 16) {
                ForEach(Array(abilityMap.domains.keys.sorted(by: { 
                    abilityMap.domains[$0]?.averageAccuracy ?? 0 > abilityMap.domains[$1]?.averageAccuracy ?? 0 
                })), id: \.self) { domain in
                    if let profile = abilityMap.domains[domain] {
                        DomainCard(
                            domain: domain,
                            profile: profile,
                            isSelected: selectedDomain == domain,
                            onTap: { selectedDomain = domain }
                        )
                    }
                }
            }
        }
    }
    
    private var recentValidationsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Recent Validations")
                .font(.title2)
                .fontWeight(.semibold)
            
            ForEach(engine.recentValidations.prefix(3), id: \.domain) { validation in
                ValidationCard(validation: validation)
            }
        }
    }
    
    private var confidenceBuilderSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Build Confidence")
                .font(.title2)
                .fontWeight(.semibold)
            
            HStack(spacing: 16) {
                DomainPicker(selectedDomain: $selectedDomain)
                ContextPicker(selectedContext: $selectedContext)
            }
            
            if let domain = selectedDomain, let context = selectedContext {
                Button(action: {
                    Task {
                        let strategy = await engine.buildContextualConfidence(
                            domain: domain,
                            context: context
                        )
                        currentStrategy = strategy
                        showingConfidenceStrategy = true
                    }
                }) {
                    HStack {
                        Image(systemName: "lightbulb.fill")
                        Text("Get Confidence Strategy")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
            }
        }
    }
    
    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("No Ability Data Yet")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Start recording your performance events to build your selective ability profile")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
            
            Button(action: { showingAddEvent = true }) {
                Text("Add First Event")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
        .padding()
    }
}

// MARK: - Supporting Views

struct OverallConfidenceView: View {
    let abilityMap: SelectiveAbilityRecognitionEngine.AbilityMap
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Overall Confidence")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text("\(Int(abilityMap.overallConfidence * 100))%")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(confidenceColor)
            }
            
            Spacer()
            
            CircularProgressView(
                progress: abilityMap.overallConfidence,
                color: confidenceColor
            )
            .frame(width: 40, height: 40)
        }
    }
    
    private var confidenceColor: Color {
        switch abilityMap.overallConfidence {
        case 0.8...:
            return .green
        case 0.5..<0.8:
            return .orange
        default:
            return .red
        }
    }
}

struct DomainCard: View {
    let domain: SelectiveAbilityRecognitionEngine.AbilityDomain
    let profile: SelectiveAbilityRecognitionEngine.PerformanceProfile
    let isSelected: Bool
    let onTap: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: domainIcon)
                    .foregroundColor(.blue)
                
                Spacer()
                
                Text("\(Int(profile.averageAccuracy * 100))%")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(accuracyColor)
            }
            
            Text(domain.description)
                .font(.headline)
                .lineLimit(2)
            
            Text("\(profile.evidenceCount) events")
                .font(.caption)
                .foregroundColor(.secondary)
            
            // Optimal contexts indicator
            if !profile.optimalConditions.isEmpty {
                HStack {
                    Image(systemName: "star.fill")
                        .font(.caption2)
                        .foregroundColor(.yellow)
                    
                    Text("Best in: \(profile.optimalConditions.first?.contextType.description ?? "")")
                        .font(.caption2)
                        .lineLimit(1)
                }
            }
        }
        .padding()
        .background(isSelected ? Color.blue.opacity(0.2) : Color.gray.opacity(0.1))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
        )
        .cornerRadius(12)
        .onTapGesture(perform: onTap)
    }
    
    private var domainIcon: String {
        switch domain {
        case .meaningfulSequences:
            return "link"
        case .pressureSituations:
            return "bolt.fill"
        case .emotionalMemories:
            return "heart.fill"
        case .routineInformation:
            return "calendar"
        case .proceduralMemory:
            return "gearshape.fill"
        case .spatialNavigation:
            return "map.fill"
        case .socialContexts:
            return "person.2.fill"
        case .temporalSequencing:
            return "clock.fill"
        }
    }
    
    private var accuracyColor: Color {
        switch profile.averageAccuracy {
        case 0.8...:
            return .green
        case 0.5..<0.8:
            return .orange
        default:
            return .red
        }
    }
}

struct ValidationCard: View {
    let validation: SelectiveAbilityRecognitionEngine.SelectiveValidation
    
    var body: some View {
        HStack {
            Image(systemName: validation.validated ? "checkmark.circle.fill" : "questionmark.circle.fill")
                .foregroundColor(validation.validated ? .green : .orange)
                .font(.title2)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(validation.domain.description)
                    .font(.headline)
                
                Text(validation.validated ? "Validated" : "Needs More Evidence")
                    .font(.caption)
                    .foregroundColor(validation.validated ? .green : .orange)
                
                Text("Confidence: \(Int(validation.confidence * 100))%")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text("\(validation.evidence.count)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(10)
    }
}

struct DomainPicker: View {
    @Binding var selectedDomain: SelectiveAbilityRecognitionEngine.AbilityDomain?
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("Domain")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Picker("Domain", selection: $selectedDomain) {
                Text("Select Domain").tag(SelectiveAbilityRecognitionEngine.AbilityDomain?.none)
                
                ForEach(SelectiveAbilityRecognitionEngine.AbilityDomain.allCases, id: \.self) { domain in
                    Text(domain.description).tag(SelectiveAbilityRecognitionEngine.AbilityDomain?.some(domain))
                }
            }
            .pickerStyle(MenuPickerStyle())
        }
    }
}

struct ContextPicker: View {
    @Binding var selectedContext: SelectiveAbilityRecognitionEngine.ContextType?
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("Context")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Picker("Context", selection: $selectedContext) {
                Text("Select Context").tag(SelectiveAbilityRecognitionEngine.ContextType?.none)
                
                ForEach(SelectiveAbilityRecognitionEngine.ContextType.allCases, id: \.self) { context in
                    Text(context.description).tag(SelectiveAbilityRecognitionEngine.ContextType?.some(context))
                }
            }
            .pickerStyle(MenuPickerStyle())
        }
    }
}

struct CircularProgressView: View {
    let progress: Double
    let color: Color
    
    var body: some View {
        ZStack {
            Circle()
                .stroke(color.opacity(0.3), lineWidth: 4)
            
            Circle()
                .trim(from: 0, to: progress)
                .stroke(color, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeInOut(duration: 0.5), value: progress)
        }
    }
}

// MARK: - Sheet Views

struct AddPerformanceEventView: View {
    @ObservedObject var engine: SelectiveAbilityRecognitionEngine
    @Environment(\.dismiss) private var dismiss
    
    @State private var selectedDomain: SelectiveAbilityRecognitionEngine.AbilityDomain = .meaningfulSequences
    @State private var selectedContext: SelectiveAbilityRecognitionEngine.ContextType = .routineTask
    @State private var accuracy: Double = 0.5
    @State private var confidence: Double = 0.5
    @State private var description: String = ""
    @State private var outcome: SelectiveAbilityRecognitionEngine.EventOutcome = .success
    
    var body: some View {
        NavigationView {
            Form {
                Section("Event Details") {
                    Picker("Domain", selection: $selectedDomain) {
                        ForEach(SelectiveAbilityRecognitionEngine.AbilityDomain.allCases, id: \.self) { domain in
                            Text(domain.description).tag(domain)
                        }
                    }
                    
                    Picker("Context", selection: $selectedContext) {
                        ForEach(SelectiveAbilityRecognitionEngine.ContextType.allCases, id: \.self) { context in
                            Text(context.description).tag(context)
                        }
                    }
                    
                    TextField("Description", text: $description, axis: .vertical)
                        .lineLimit(3)
                }
                
                Section("Performance") {
                    VStack {
                        HStack {
                            Text("Accuracy")
                            Spacer()
                            Text("\(Int(accuracy * 100))%")
                        }
                        Slider(value: $accuracy, in: 0...1)
                    }
                    
                    VStack {
                        HStack {
                            Text("Your Confidence")
                            Spacer()
                            Text("\(Int(confidence * 100))%")
                        }
                        Slider(value: $confidence, in: 0...1)
                    }
                    
                    Picker("Outcome", selection: $outcome) {
                        ForEach(SelectiveAbilityRecognitionEngine.EventOutcome.allCases, id: \.self) { outcome in
                            Text(outcome.description).tag(outcome)
                        }
                    }
                }
            }
            .navigationTitle("Add Event")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        let event = SelectiveAbilityRecognitionEngine.PerformanceEvent(
                            domain: selectedDomain,
                            context: selectedContext,
                            accuracy: accuracy,
                            confidence: confidence,
                            description: description,
                            outcome: outcome
                        )
                        
                        engine.recordPerformanceEvent(event)
                        dismiss()
                    }
                    .disabled(description.isEmpty)
                }
            }
        }
    }
}

struct ConfidenceStrategyView: View {
    let strategy: SelectiveAbilityRecognitionEngine.ConfidenceStrategy
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header
                    VStack(alignment: .leading, spacing: 8) {
                        Text(strategy.strategy.description)
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        Text(strategy.description)
                            .font(.body)
                            .foregroundColor(.secondary)
                    }
                    
                    // Implementation Steps
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Implementation Steps")
                            .font(.headline)
                        
                        ForEach(Array(strategy.implementationSteps.enumerated()), id: \.offset) { index, step in
                            HStack(alignment: .top, spacing: 12) {
                                Text("\(index + 1)")
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(.white)
                                    .frame(width: 24, height: 24)
                                    .background(Color.blue)
                                    .clipShape(Circle())
                                
                                Text(step)
                                    .font(.body)
                            }
                        }
                    }
                    
                    // Expected Outcome
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Expected Outcome")
                            .font(.headline)
                        
                        Text(strategy.expectedOutcome)
                            .font(.body)
                            .padding()
                            .background(Color.green.opacity(0.1))
                            .cornerRadius(8)
                    }
                    
                    // Risk Mitigation
                    if !strategy.riskMitigation.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Risk Mitigation")
                                .font(.headline)
                            
                            ForEach(strategy.riskMitigation, id: \.self) { risk in
                                HStack(alignment: .top) {
                                    Image(systemName: "shield.fill")
                                        .foregroundColor(.orange)
                                        .font(.caption)
                                    
                                    Text(risk)
                                        .font(.body)
                                }
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Confidence Strategy")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#Preview {
    SelectiveAbilityView()
} 