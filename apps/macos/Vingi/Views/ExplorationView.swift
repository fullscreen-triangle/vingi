import SwiftUI
import VingiCore

struct ExplorationView: View {
    @StateObject private var explorationEngine = ExplorationEngine()
    @State private var selectedCategory: ExplorationEngine.ExplorationCategory = .restaurants
    @State private var currentLocation = "Nuremberg City Center"
    @State private var showingDetails = false
    @State private var selectedSuggestion: ExplorationEngine.ExplorationSuggestion?
    
    var body: some View {
        NavigationView {
            VStack(alignment: .leading, spacing: 20) {
                headerSection
                problemStatementSection
                explorationSuggestionsSection
                Spacer()
            }
            .padding()
            .navigationTitle("Break Default Patterns")
        }
        .sheet(item: $selectedSuggestion) { suggestion in
            ExplorationDetailView(suggestion: suggestion) { result in
                explorationEngine.recordExplorationResult(result)
                selectedSuggestion = nil
            }
        }
        .task {
            await loadSuggestions()
        }
    }
    
    // MARK: - Header Section
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Expand Your Horizons Safely")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Try new things without the fear of disappointment")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            HStack {
                Text("Location:")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                TextField("Current location", text: $currentLocation)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                
                Button("Update") {
                    Task {
                        await loadSuggestions()
                    }
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    // MARK: - Problem Statement
    
    private var problemStatementSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Your Current Pattern")
                .font(.headline)
            
            if selectedCategory == .shopping {
                shoppingConstraintProblem
            } else {
                defaultBehaviorProblem
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("✅ How Vingi Helps:")
                    .fontWeight(.semibold)
                    .foregroundColor(.green)
                
                if selectedCategory == .shopping {
                    shoppingVingiSolution
                } else {
                    defaultVingiSolution
                }
            }
            .padding()
            .background(Color.green.opacity(0.1))
            .cornerRadius(8)
        }
    }
    
    private var shoppingConstraintProblem: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("🛒 The Supermarket Constraint Loop:")
                .fontWeight(.semibold)
                .foregroundColor(.orange)
            
            HStack(alignment: .top) {
                Text("•")
                Text("Optimized your preferences to fit what's at the 1-minute supermarket")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Given up on better bread, fresher milk, specialty items")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Life fits around store constraints instead of store fitting your life")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Missing bonus experiences: walks, discovery, better quality")
            }
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(8)
    }
    
    private var defaultBehaviorProblem: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("🔄 The Loop You're Stuck In:")
                .fontWeight(.semibold)
                .foregroundColor(.orange)
            
            HStack(alignment: .top) {
                Text("•")
                Text("Think \"I want a burger\" → automatically go to Five Guys")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Go to Greek restaurant → order the same things every time")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Want to try something new → don't trust ratings → stick with familiar")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Miss out on potentially better options that are right nearby")
            }
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(8)
    }
    
    private var shoppingVingiSolution: some View {
        Group {
            HStack(alignment: .top) {
                Text("•")
                Text("Optimize shopping routes instead of constraining preferences")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Same budget, same time, but better products + bonus experiences")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Your 1-minute store becomes backup, not constraint")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Strategic multi-stop routes: milk shop → bakery → main groceries")
            }
        }
    }
    
    private var defaultVingiSolution: some View {
        Group {
            HStack(alignment: .top) {
                Text("•")
                Text("Suggests alternatives with high similarity to what you already like")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Always provides a safety net - your fallback is never far away")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Uses quality indicators you actually trust")
            }
            
            HStack(alignment: .top) {
                Text("•")
                Text("Gradual expansion of comfort zone without overwhelming choices")
            }
        }
    }
    
    // MARK: - Exploration Suggestions
    
    private var explorationSuggestionsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Smart Alternatives")
                    .font(.headline)
                
                Spacer()
                
                if explorationEngine.isGenerating {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            
            categorySelector
            
            if explorationEngine.currentSuggestions.isEmpty && !explorationEngine.isGenerating {
                emptyStateView
            } else {
                suggestionsList
            }
        }
    }
    
    private var categorySelector: some View {
        HStack {
            Text("Category:")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Picker("Category", selection: $selectedCategory) {
                Text("Restaurants").tag(ExplorationEngine.ExplorationCategory.restaurants)
                Text("Menu Items").tag(ExplorationEngine.ExplorationCategory.food_items)
                Text("Areas").tag(ExplorationEngine.ExplorationCategory.neighborhoods)
                Text("Shopping").tag(ExplorationEngine.ExplorationCategory.shopping)
            }
            .pickerStyle(SegmentedPickerStyle())
            .onChange(of: selectedCategory) { _ in
                Task {
                    await loadSuggestions()
                }
            }
        }
    }
    
    private var suggestionsList: some View {
        VStack(spacing: 12) {
            ForEach(explorationEngine.currentSuggestions) { suggestion in
                suggestionCard(suggestion)
            }
        }
    }
    
    private func suggestionCard(_ suggestion: ExplorationEngine.ExplorationSuggestion) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with risk level
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(suggestion.newOption.name)
                        .font(.headline)
                    
                    Text(suggestion.newOption.description)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                riskBadge(suggestion.riskLevel)
            }
            
            // Shopping-specific benefits analysis
            if suggestion.category == .shopping {
                shoppingBenefitsView
            }
            
            // Similarity indicator
            similarityIndicator(suggestion.newOption.similarityToKnownPreferences)
            
            // Quality indicators
            qualityIndicatorsView(suggestion.newOption.qualityIndicators)
            
            // Specific recommendation
            if let specificRec = suggestion.newOption.specificRecommendation {
                HStack(alignment: .top) {
                    Image(systemName: "lightbulb")
                        .foregroundColor(.blue)
                        .font(.caption)
                    
                    Text(specificRec)
                        .font(.caption)
                        .foregroundColor(.blue)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.blue.opacity(0.1))
                .cornerRadius(6)
            }
            
            // Safety net summary
            safetyNetSummary(suggestion.safetyNet)
            
            // Action buttons
            HStack {
                Button("See Details") {
                    selectedSuggestion = suggestion
                }
                .buttonStyle(.bordered)
                
                Spacer()
                
                Button("Try This") {
                    selectedSuggestion = suggestion
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var shoppingBenefitsView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Benefits of Multi-Stop Strategy:")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.blue)
            
            let benefits = explorationEngine.calculateShoppingBenefits(multiStopRoute: true)
            
            HStack(alignment: .top) {
                Image(systemName: "figure.walk")
                    .foregroundColor(.green)
                Text("Exercise: \(benefits.extraWalkingTime)")
                    .font(.caption)
            }
            
            HStack(alignment: .top) {
                Image(systemName: "eye")
                    .foregroundColor(.purple)
                Text("Discovery: \(benefits.discoveryOpportunities)")
                    .font(.caption)
            }
            
            HStack(alignment: .top) {
                Image(systemName: "star")
                    .foregroundColor(.orange)
                Text("Quality: \(benefits.productQualityImprovement)")
                    .font(.caption)
            }
            
            HStack(alignment: .top) {
                Image(systemName: "person.2")
                    .foregroundColor(.blue)
                Text("Social: \(benefits.socialInteractions)")
                    .font(.caption)
            }
        }
        .padding(8)
        .background(Color.blue.opacity(0.05))
        .cornerRadius(6)
    }
    
    private func riskBadge(_ risk: ExplorationEngine.RiskLevel) -> some View {
        Text(risk.rawValue.uppercased())
            .font(.caption2)
            .fontWeight(.semibold)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(riskColor(risk))
            .foregroundColor(.white)
            .cornerRadius(4)
    }
    
    private func riskColor(_ risk: ExplorationEngine.RiskLevel) -> Color {
        switch risk {
        case .minimal: return .green
        case .low: return .blue
        case .moderate: return .orange
        case .high: return .red
        }
    }
    
    private func similarityIndicator(_ similarity: Double) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Similarity to your tastes:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text("\(Int(similarity * 100))%")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(similarityColor(similarity))
            }
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color(.systemGray5))
                        .frame(height: 4)
                    
                    Rectangle()
                        .fill(similarityColor(similarity))
                        .frame(width: geometry.size.width * similarity, height: 4)
                }
            }
            .frame(height: 4)
            .cornerRadius(2)
        }
    }
    
    private func similarityColor(_ similarity: Double) -> Color {
        if similarity >= 0.8 { return .green }
        else if similarity >= 0.6 { return .blue }
        else if similarity >= 0.4 { return .orange }
        else { return .red }
    }
    
    private func qualityIndicatorsView(_ indicators: [ExplorationEngine.QualityIndicator]) -> some View {
        LazyVGrid(columns: [
            GridItem(.adaptive(minimum: 100))
        ], spacing: 4) {
            ForEach(Array(indicators.prefix(4)), id: \.self) { indicator in
                Text(indicatorText(indicator))
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(4)
            }
        }
    }
    
    private func indicatorText(_ indicator: ExplorationEngine.QualityIndicator) -> String {
        switch indicator {
        case .michelin_recommended: return "Michelin ⭐"
        case .local_institution: return "Local Favorite"
        case .chef_owned: return "Chef-Owned"
        case .fresh_ingredients: return "Fresh Ingredients"
        case .traditional_methods: return "Traditional"
        case .popular_with_locals: return "Popular Locally"
        case .similar_to_known_favorite: return "Like Your Usual"
        case .verified_good_reviews: return "Good Reviews"
        case .long_established: return "Established"
        }
    }
    
    private func safetyNetSummary(_ safetyNet: ExplorationEngine.SafetyNet) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Safety Net:")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)
            
            HStack {
                Image(systemName: "shield")
                    .font(.caption)
                    .foregroundColor(.green)
                
                Text("Fallback: \(safetyNet.fallbackDistance)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text("Time: \(safetyNet.timeCommitment)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(8)
        .background(Color.green.opacity(0.05))
        .cornerRadius(6)
    }
    
    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "map")
                .font(.system(size: 48))
                .foregroundColor(.gray)
            
            Text("No suggestions available")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("Try a different category or location")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Button("Refresh Suggestions") {
                Task {
                    await loadSuggestions()
                }
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }
    
    // MARK: - Helper Methods
    
    private func loadSuggestions() async {
        switch selectedCategory {
        case .restaurants:
            await explorationEngine.suggestNurembergDining(currentLocation: currentLocation)
        case .food_items:
            let suggestion = await explorationEngine.suggestMenuExploration(
                restaurant: "Greek restaurant",
                usualOrder: ["Usual gyros", "Regular drink"]
            )
            DispatchQueue.main.async {
                self.explorationEngine.currentSuggestions = [suggestion]
            }
        case .neighborhoods:
            let suggestion = await explorationEngine.suggestLocationExploration(currentArea: currentLocation)
            DispatchQueue.main.async {
                self.explorationEngine.currentSuggestions = [suggestion]
            }
        case .shopping:
            let suggestion = await explorationEngine.optimizeShoppingStrategy(
                currentStore: "Nearby supermarket",
                currentCommute: "1 minute walk",
                weeklySpend: 50.0,
                constrainedPreferences: ["Store-brand bread", "Limited produce selection"]
            )
            DispatchQueue.main.async {
                self.explorationEngine.currentSuggestions = [suggestion]
            }
        default:
            break
        }
    }
}

// MARK: - Exploration Detail View

struct ExplorationDetailView: View {
    let suggestion: ExplorationEngine.ExplorationSuggestion
    let onResult: (ExplorationEngine.ExplorationResult) -> Void
    
    @Environment(\.dismiss) private var dismiss
    @State private var userRating: Double = 3.0
    @State private var outcome: ExplorationEngine.ExplorationOutcome = .neutral
    @State private var notes: String = ""
    @State private var wouldRecommend: Bool = true
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Main suggestion details
                    suggestionDetailsSection
                    
                    // Fallback option
                    fallbackSection
                    
                    // Safety net details
                    safetyNetSection
                    
                    // Feedback section
                    feedbackSection
                }
                .padding()
            }
            .navigationTitle("Exploration Plan")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Record Result") {
                        recordResult()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
    }
    
    private var suggestionDetailsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Try This:")
                .font(.headline)
            
            VStack(alignment: .leading, spacing: 8) {
                Text(suggestion.newOption.name)
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text(suggestion.newOption.description)
                    .font(.body)
                    .foregroundColor(.secondary)
                
                if let location = suggestion.newOption.location.isEmpty ? nil : suggestion.newOption.location {
                    HStack {
                        Image(systemName: "location")
                        Text(location)
                            .font(.subheadline)
                    }
                    .foregroundColor(.blue)
                }
                
                if suggestion.newOption.estimatedCost > 0 {
                    HStack {
                        Image(systemName: "eurosign.circle")
                        Text("~€\(String(format: "%.0f", suggestion.newOption.estimatedCost))")
                            .font(.subheadline)
                    }
                    .foregroundColor(.green)
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
            
            if let specificRec = suggestion.newOption.specificRecommendation {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: "lightbulb.fill")
                            .foregroundColor(.orange)
                        Text("Specific Recommendation")
                            .fontWeight(.semibold)
                    }
                    
                    Text(specificRec)
                        .font(.body)
                }
                .padding()
                .background(Color.orange.opacity(0.1))
                .cornerRadius(12)
            }
        }
    }
    
    private var fallbackSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Your Safety Net:")
                .font(.headline)
            
            VStack(alignment: .leading, spacing: 8) {
                Text(suggestion.fallbackOption.name)
                    .font(.title3)
                    .fontWeight(.semibold)
                
                Text(suggestion.fallbackOption.description)
                    .font(.body)
                    .foregroundColor(.secondary)
                
                Text("Distance: \(suggestion.safetyNet.fallbackDistance)")
                    .font(.subheadline)
                    .foregroundColor(.blue)
            }
            .padding()
            .background(Color.green.opacity(0.1))
            .cornerRadius(12)
        }
    }
    
    private var safetyNetSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Exit Strategy:")
                .font(.headline)
            
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top) {
                    Image(systemName: "shield.fill")
                        .foregroundColor(.green)
                    Text(suggestion.safetyNet.exitStrategy)
                }
                
                HStack(alignment: .top) {
                    Image(systemName: "clock")
                        .foregroundColor(.blue)
                    Text("Time commitment: \(suggestion.safetyNet.timeCommitment)")
                }
                
                HStack(alignment: .top) {
                    Image(systemName: "eurosign.circle")
                        .foregroundColor(.orange)
                    Text("Cost consideration: \(suggestion.safetyNet.costLimit)")
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
        }
    }
    
    private var feedbackSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("How did it go?")
                .font(.headline)
            
            VStack(spacing: 16) {
                // Outcome picker
                VStack(alignment: .leading, spacing: 8) {
                    Text("Outcome:")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    
                    Picker("Outcome", selection: $outcome) {
                        Text("Loved it! 😍").tag(ExplorationEngine.ExplorationOutcome.loved_it)
                        Text("Liked it 👍").tag(ExplorationEngine.ExplorationOutcome.liked_it)
                        Text("It was okay 😐").tag(ExplorationEngine.ExplorationOutcome.neutral)
                        Text("Disappointed 😞").tag(ExplorationEngine.ExplorationOutcome.disappointed)
                        Text("Used fallback").tag(ExplorationEngine.ExplorationOutcome.used_fallback)
                        Text("Gave up").tag(ExplorationEngine.ExplorationOutcome.abandoned)
                    }
                    .pickerStyle(MenuPickerStyle())
                }
                
                // Rating
                VStack(alignment: .leading, spacing: 8) {
                    Text("Rating: \(String(format: "%.1f", userRating))/5.0")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    
                    Slider(value: $userRating, in: 1...5, step: 0.5)
                }
                
                // Notes
                VStack(alignment: .leading, spacing: 8) {
                    Text("Notes (optional):")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    
                    TextField("What worked or didn't work?", text: $notes, axis: .vertical)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .lineLimit(3...6)
                }
                
                // Would recommend
                Toggle("Would recommend to others like me", isOn: $wouldRecommend)
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
        }
    }
    
    private func recordResult() {
        let result = ExplorationEngine.ExplorationResult(
            suggestion: suggestion,
            outcome: outcome,
            userRating: userRating,
            notes: notes.isEmpty ? nil : notes,
            wouldRecommendToOthers: wouldRecommend
        )
        
        onResult(result)
    }
}

#Preview {
    ExplorationView()
} 