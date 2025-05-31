import Foundation
import Combine

/// Engine that helps break default behavior patterns while minimizing risk and cognitive load
/// Specifically designed for "I always go to Five Guys even though I want to try new places" scenarios
public class ExplorationEngine: ObservableObject {
    
    // MARK: - Types
    
    public struct ExplorationSuggestion: Identifiable, Codable {
        public let id: UUID
        public let category: ExplorationCategory
        public let newOption: RecommendedOption
        public let fallbackOption: RecommendedOption
        public let riskLevel: RiskLevel
        public let confidence: Double
        public let reasoning: String
        public let safetyNet: SafetyNet
        
        public init(
            id: UUID = UUID(),
            category: ExplorationCategory,
            newOption: RecommendedOption,
            fallbackOption: RecommendedOption,
            riskLevel: RiskLevel,
            confidence: Double,
            reasoning: String,
            safetyNet: SafetyNet
        ) {
            self.id = id
            self.category = category
            self.newOption = newOption
            self.fallbackOption = fallbackOption
            self.riskLevel = riskLevel
            self.confidence = confidence
            self.reasoning = reasoning
            self.safetyNet = safetyNet
        }
    }
    
    public struct RecommendedOption: Codable {
        public let name: String
        public let description: String
        public let location: String
        public let estimatedCost: Double
        public let similarityToKnownPreferences: Double // 0-1, higher = more similar to what they like
        public let qualityIndicators: [QualityIndicator]
        public let specificRecommendation: String? // "Try the lamb gyros - similar to your usual but better"
        
        public init(
            name: String,
            description: String,
            location: String,
            estimatedCost: Double,
            similarityToKnownPreferences: Double,
            qualityIndicators: [QualityIndicator],
            specificRecommendation: String? = nil
        ) {
            self.name = name
            self.description = description
            self.location = location
            self.estimatedCost = estimatedCost
            self.similarityToKnownPreferences = similarityToKnownPreferences
            self.qualityIndicators = qualityIndicators
            self.specificRecommendation = specificRecommendation
        }
    }
    
    public enum ExplorationCategory: String, CaseIterable, Codable {
        case restaurants = "restaurants"
        case food_items = "food_items"        // Different dishes at same restaurant
        case neighborhoods = "neighborhoods"
        case activities = "activities"
        case shopping = "shopping"
        case entertainment = "entertainment"
    }
    
    public enum RiskLevel: String, CaseIterable, Codable {
        case minimal = "minimal"     // 95%+ chance you'll like it
        case low = "low"            // 85%+ chance
        case moderate = "moderate"   // 70%+ chance
        case high = "high"          // 50%+ chance, but worth trying
    }
    
    public enum QualityIndicator: String, CaseIterable, Codable {
        case michelin_recommended = "michelin_recommended"
        case local_institution = "local_institution"
        case chef_owned = "chef_owned"
        case fresh_ingredients = "fresh_ingredients"
        case traditional_methods = "traditional_methods"
        case popular_with_locals = "popular_with_locals"
        case similar_to_known_favorite = "similar_to_known_favorite"
        case verified_good_reviews = "verified_good_reviews"
        case long_established = "long_established"
    }
    
    public struct SafetyNet: Codable {
        public let exitStrategy: String
        public let timeCommitment: String
        public let costLimit: String
        public let fallbackDistance: String
        
        public init(
            exitStrategy: String,
            timeCommitment: String,
            costLimit: String,
            fallbackDistance: String
        ) {
            self.exitStrategy = exitStrategy
            self.timeCommitment = timeCommitment
            self.costLimit = costLimit
            self.fallbackDistance = fallbackDistance
        }
    }
    
    public struct ShoppingBenefits: Codable {
        public let extraWalkingTime: String
        public let discoveryOpportunities: String
        public let productQualityImprovement: String
        public let socialInteractions: String
        public let flexibilityIncrease: String
        public let costOptimization: String
        
        public init(
            extraWalkingTime: String,
            discoveryOpportunities: String,
            productQualityImprovement: String,
            socialInteractions: String,
            flexibilityIncrease: String,
            costOptimization: String
        ) {
            self.extraWalkingTime = extraWalkingTime
            self.discoveryOpportunities = discoveryOpportunities
            self.productQualityImprovement = productQualityImprovement
            self.socialInteractions = socialInteractions
            self.flexibilityIncrease = flexibilityIncrease
            self.costOptimization = costOptimization
        }
    }
    
    public struct UserPattern: Codable {
        public let knownPreferences: [String]
        public let avoidances: [String]
        public let comfortZone: [String]
        public let lastExplorations: [ExplorationResult]
        public let riskTolerance: RiskLevel
        
        public init(
            knownPreferences: [String],
            avoidances: [String] = [],
            comfortZone: [String],
            lastExplorations: [ExplorationResult] = [],
            riskTolerance: RiskLevel = .low
        ) {
            self.knownPreferences = knownPreferences
            self.avoidances = avoidances
            self.comfortZone = comfortZone
            self.lastExplorations = lastExplorations
            self.riskTolerance = riskTolerance
        }
    }
    
    public struct ExplorationResult: Codable {
        public let suggestion: ExplorationSuggestion
        public let outcome: ExplorationOutcome
        public let userRating: Double
        public let notes: String?
        public let wouldRecommendToOthers: Bool
        
        public init(
            suggestion: ExplorationSuggestion,
            outcome: ExplorationOutcome,
            userRating: Double,
            notes: String? = nil,
            wouldRecommendToOthers: Bool
        ) {
            self.suggestion = suggestion
            self.outcome = outcome
            self.userRating = userRating
            self.notes = notes
            self.wouldRecommendToOthers = wouldRecommendToOthers
        }
    }
    
    public enum ExplorationOutcome: String, CaseIterable, Codable {
        case loved_it = "loved_it"
        case liked_it = "liked_it"
        case neutral = "neutral"
        case disappointed = "disappointed"
        case used_fallback = "used_fallback"
        case abandoned = "abandoned"
    }
    
    // MARK: - Properties
    
    @Published public private(set) var currentSuggestions: [ExplorationSuggestion] = []
    @Published public private(set) var isGenerating = false
    
    private var userPattern: UserPattern
    private var explorationHistory: [ExplorationResult] = []
    
    // MARK: - Initialization
    
    public init(userPattern: UserPattern = UserPattern.defaultPattern) {
        self.userPattern = userPattern
    }
    
    // MARK: - Public Methods
    
    /// Generate exploration suggestions for Nuremberg dining
    /// Specifically addresses "I only know Five Guys but want to try new things"
    public func suggestNurembergDining(currentLocation: String = "Nuremberg City Center") async -> [ExplorationSuggestion] {
        isGenerating = true
        defer { isGenerating = false }
        
        let suggestions = await generateNurembergDiningSuggestions()
        
        DispatchQueue.main.async {
            self.currentSuggestions = suggestions
        }
        
        return suggestions
    }
    
    /// Help break the "same order at familiar restaurant" pattern
    public func suggestMenuExploration(restaurant: String, usualOrder: [String]) async -> ExplorationSuggestion {
        
        // For your Greek restaurant example
        if restaurant.lowercased().contains("greek") {
            return ExplorationSuggestion(
                category: .food_items,
                newOption: RecommendedOption(
                    name: "Lamb Kleftiko",
                    description: "Slow-cooked lamb with herbs - richer version of what you usually get",
                    location: "Same Greek restaurant",
                    estimatedCost: 0, // No extra cost, just different dish
                    similarityToKnownPreferences: 0.85,
                    qualityIndicators: [.traditional_methods, .similar_to_known_favorite],
                    specificRecommendation: "Ask for it prepared the traditional way - it's like your usual but more complex flavors"
                ),
                fallbackOption: RecommendedOption(
                    name: "Your usual order",
                    description: "What you always get",
                    location: "Same table",
                    estimatedCost: 0,
                    similarityToKnownPreferences: 1.0,
                    qualityIndicators: [.similar_to_known_favorite]
                ),
                riskLevel: .minimal,
                confidence: 0.9,
                reasoning: "Since you like your usual Greek dishes, this is a natural extension with similar flavors but more depth",
                safetyNet: SafetyNet(
                    exitStrategy: "Order your usual as a side if you don't like the new dish",
                    timeCommitment: "Same as normal visit",
                    costLimit: "Maybe €3-5 more than usual",
                    fallbackDistance: "Zero - you're already in your comfort zone restaurant"
                )
            )
        }
        
        // Generic suggestion for unknown restaurants
        return createGenericMenuExplorationSuggestion(usualOrder: usualOrder)
    }
    
    /// Break location defaults - "I always go to the same areas in Nuremberg"
    public func suggestLocationExploration(currentArea: String = "City Center") async -> ExplorationSuggestion {
        
        if currentArea.lowercased().contains("center") || currentArea.lowercased().contains("hauptbahnhof") {
            return ExplorationSuggestion(
                category: .neighborhoods,
                newOption: RecommendedOption(
                    name: "Gostenhof District",
                    description: "Hip neighborhood with authentic local spots, 10 minutes from center",
                    location: "U-Bahn to Gostenhof",
                    estimatedCost: 3.0, // U-Bahn ticket
                    similarityToKnownPreferences: 0.7,
                    qualityIndicators: [.popular_with_locals, .long_established],
                    specificRecommendation: "Start with coffee at a local cafe, then explore from there"
                ),
                fallbackOption: RecommendedOption(
                    name: "City Center",
                    description: "Your usual area with Five Guys etc.",
                    location: "Hauptmarkt area",
                    estimatedCost: 0,
                    similarityToKnownPreferences: 1.0,
                    qualityIndicators: [.similar_to_known_favorite]
                ),
                riskLevel: .low,
                confidence: 0.8,
                reasoning: "Easy to get to, easy to get back, but gives you access to places locals actually go",
                safetyNet: SafetyNet(
                    exitStrategy: "U-Bahn back to center anytime (10 minutes)",
                    timeCommitment: "Can explore for 30 minutes or all day",
                    costLimit: "Just transport cost",
                    fallbackDistance: "10 minutes back to familiar territory"
                )
            )
        }
        
        return createGenericLocationSuggestion()
    }
    
    /// Break the "single supermarket constraint" - optimize shopping routes instead of constraining preferences
    public func optimizeShoppingStrategy(
        currentStore: String = "Nearby supermarket",
        currentCommute: String = "1 minute walk",
        weeklySpend: Double = 50.0,
        constrainedPreferences: [String] = []
    ) async -> ExplorationSuggestion {
        
        // This addresses the pattern: "I only buy what's at the close supermarket"
        // Instead: "I buy what I actually want, optimizing the route to get it efficiently"
        
        if currentCommute.contains("1 minute") || currentCommute.contains("very close") {
            return ExplorationSuggestion(
                category: .shopping,
                newOption: RecommendedOption(
                    name: "Multi-Stop Shopping Route",
                    description: "Strategic shopping across 2-3 locations for better products + bonus experiences",
                    location: "Planned route: Milk → Bakery → Main groceries",
                    estimatedCost: weeklySpend, // Same total budget
                    similarityToKnownPreferences: 0.9, // Better products you actually want
                    qualityIndicators: [.fresh_ingredients, .local_institution, .verified_good_reviews],
                    specificRecommendation: "Monday: Fresh milk from nearby dairy shop (3 min walk), grab newspaper, notice neighborhood. Wednesday: Fresh bread from actual bakery (8 min walk), discover new coffee place next door. Saturday: Main grocery run at larger store (15 min walk), better selection, same brands you like."
                ),
                fallbackOption: RecommendedOption(
                    name: currentStore,
                    description: "Your current 1-minute supermarket strategy",
                    location: currentCommute,
                    estimatedCost: weeklySpend,
                    similarityToKnownPreferences: 0.6, // Limited by their selection
                    qualityIndicators: [.similar_to_known_favorite]
                ),
                riskLevel: .minimal,
                confidence: 0.92,
                reasoning: "You're spending the same money and time, but getting better products AND bonus experiences. The 1-minute place becomes your emergency backup, not your constraint.",
                safetyNet: SafetyNet(
                    exitStrategy: "Your 1-minute supermarket is always there for anything you can't find or if you're in a rush",
                    timeCommitment: "Actually adds valuable 'outside time' to your week",
                    costLimit: "Same weekly budget, just distributed differently",
                    fallbackDistance: "1-minute backup always available"
                )
            )
        }
        
        return createGenericShoppingSuggestion(weeklySpend: weeklySpend)
    }
    
    /// Analyze what preferences have been constrained by current shopping habits
    public func identifyConstrainedPreferences(currentStore: String) -> [String] {
        // This would analyze what the user might actually prefer if not limited by one store's selection
        return [
            "Better bread (real bakery vs packaged)",
            "Fresher produce (market vs supermarket)",
            "Local dairy products",
            "Specialty items you've given up on",
            "Seasonal/rotating selections"
        ]
    }
    
    /// Calculate the hidden benefits of distributed shopping
    public func calculateShoppingBenefits(multiStopRoute: Bool) -> ShoppingBenefits {
        if multiStopRoute {
            return ShoppingBenefits(
                extraWalkingTime: "15-20 minutes more per week",
                discoveryOpportunities: "2-3 new places per month",
                productQualityImprovement: "Fresher bread, milk, produce",
                socialInteractions: "Real shopkeepers vs self-checkout",
                flexibilityIncrease: "Not dependent on one store's stock/hours",
                costOptimization: "Better prices on specific items at specialist shops"
            )
        } else {
            return ShoppingBenefits(
                extraWalkingTime: "Minimal",
                discoveryOpportunities: "None - same route always",
                productQualityImprovement: "Limited by single store's selection",
                socialInteractions: "Minimal",
                flexibilityIncrease: "Low - single point of failure",
                costOptimization: "Convenience premium on all items"
            )
        }
    }
    
    // MARK: - Private Methods
    
    private func generateNurembergDiningSuggestions() async -> [ExplorationSuggestion] {
        
        // Burger alternatives to Five Guys
        let burgerAlternative = ExplorationSuggestion(
            category: .restaurants,
            newOption: RecommendedOption(
                name: "Hans im Glück",
                description: "German burger chain with fresh ingredients and unique combinations",
                location: "Königstraße 74, 5 minutes from Five Guys",
                estimatedCost: 12.0,
                similarityToKnownPreferences: 0.9, // Very similar to Five Guys
                qualityIndicators: [.fresh_ingredients, .popular_with_locals, .verified_good_reviews],
                specificRecommendation: "Try the 'Truthahn' (turkey) burger - quality is as good as Five Guys but different flavors"
            ),
            fallbackOption: RecommendedOption(
                name: "Five Guys",
                description: "Your reliable burger choice",
                location: "Your usual spot",
                estimatedCost: 15.0,
                similarityToKnownPreferences: 1.0,
                qualityIndicators: [.similar_to_known_favorite]
            ),
            riskLevel: .minimal,
            confidence: 0.95,
            reasoning: "Similar concept to Five Guys (fresh burgers) but German approach. If you don't like it, Five Guys is 2 minutes walk away.",
            safetyNet: SafetyNet(
                exitStrategy: "Five Guys is literally around the corner",
                timeCommitment: "Same as Five Guys visit",
                costLimit: "Actually slightly cheaper than Five Guys",
                fallbackDistance: "2-minute walk to Five Guys"
            )
        )
        
        // Traditional Franconian food (safe but different)
        let traditionalOption = ExplorationSuggestion(
            category: .restaurants,
            newOption: RecommendedOption(
                name: "Hausbrauerei Altstadthof",
                description: "Traditional Franconian brewery restaurant, tourist-friendly but authentic",
                location: "Bergstraße 19, Old Town",
                estimatedCost: 18.0,
                similarityToKnownPreferences: 0.6,
                qualityIndicators: [.local_institution, .traditional_methods, .long_established],
                specificRecommendation: "Try the Schäuferla (roasted pork shoulder) - it's Nuremberg's signature dish and hard to mess up"
            ),
            fallbackOption: RecommendedOption(
                name: "Five Guys",
                description: "Your reliable burger choice",
                location: "Hauptbahnhof area",
                estimatedCost: 15.0,
                similarityToKnownPreferences: 1.0,
                qualityIndicators: [.similar_to_known_favorite]
            ),
            riskLevel: .low,
            confidence: 0.8,
            reasoning: "This is THE place tourists and locals go for traditional Nuremberg food. Quality is consistent and it's an experience, not just a meal.",
            safetyNet: SafetyNet(
                exitStrategy: "Lots of other restaurants in Old Town if you hate it",
                timeCommitment: "1-2 hours for full experience",
                costLimit: "More expensive but you're paying for the experience",
                fallbackDistance: "15-minute walk to Five Guys"
            )
        )
        
        // International but safe option
        let internationalOption = ExplorationSuggestion(
            category: .restaurants,
            newOption: RecommendedOption(
                name: "Vapiano",
                description: "Italian fast-casual chain, similar concept to Five Guys but pasta/pizza",
                location: "Königstraße, near shopping area",
                estimatedCost: 12.0,
                similarityToKnownPreferences: 0.8,
                qualityIndicators: [.fresh_ingredients, .popular_with_locals],
                specificRecommendation: "Try the Aglio e Olio pasta - simple, hard to mess up, and you watch them make it"
            ),
            fallbackOption: RecommendedOption(
                name: "Five Guys",
                description: "Your reliable burger choice",
                location: "Nearby",
                estimatedCost: 15.0,
                similarityToKnownPreferences: 1.0,
                qualityIndicators: [.similar_to_known_favorite]
            ),
            riskLevel: .minimal,
            confidence: 0.9,
            reasoning: "Same fast-casual concept as Five Guys, just Italian food. You can see them making your food, so quality is predictable.",
            safetyNet: SafetyNet(
                exitStrategy: "Quick service, so if you don't like it you're not stuck long",
                timeCommitment: "30-45 minutes max",
                costLimit: "Similar to Five Guys pricing",
                fallbackDistance: "5-minute walk to Five Guys"
            )
        )
        
        return [burgerAlternative, internationalOption, traditionalOption]
    }
    
    private func createGenericMenuExplorationSuggestion(usualOrder: [String]) -> ExplorationSuggestion {
        return ExplorationSuggestion(
            category: .food_items,
            newOption: RecommendedOption(
                name: "Chef's recommendation",
                description: "Ask server what's similar to your usual but different",
                location: "Same restaurant",
                estimatedCost: 0,
                similarityToKnownPreferences: 0.7,
                qualityIndicators: [.similar_to_known_favorite]
            ),
            fallbackOption: RecommendedOption(
                name: "Your usual order",
                description: "What you always get",
                location: "Same table",
                estimatedCost: 0,
                similarityToKnownPreferences: 1.0,
                qualityIndicators: [.similar_to_known_favorite]
            ),
            riskLevel: .low,
            confidence: 0.7,
            reasoning: "Server knows what's good and similar to what you usually like",
            safetyNet: SafetyNet(
                exitStrategy: "Order your usual as backup",
                timeCommitment: "Same as normal",
                costLimit: "Similar price range",
                fallbackDistance: "Zero risk - same restaurant"
            )
        )
    }
    
    private func createGenericLocationSuggestion() -> ExplorationSuggestion {
        return ExplorationSuggestion(
            category: .neighborhoods,
            newOption: RecommendedOption(
                name: "Adjacent area",
                description: "Explore one street over from your usual spot",
                location: "Walking distance from comfort zone",
                estimatedCost: 0,
                similarityToKnownPreferences: 0.8,
                qualityIndicators: [.similar_to_known_favorite]
            ),
            fallbackOption: RecommendedOption(
                name: "Usual area",
                description: "Your familiar territory",
                location: "Where you always go",
                estimatedCost: 0,
                similarityToKnownPreferences: 1.0,
                qualityIndicators: [.similar_to_known_favorite]
            ),
            riskLevel: .minimal,
            confidence: 0.8,
            reasoning: "Minimal exploration with easy retreat to familiar area",
            safetyNet: SafetyNet(
                exitStrategy: "Walk back to familiar area",
                timeCommitment: "As long or short as you want",
                costLimit: "No extra cost",
                fallbackDistance: "1-2 minutes walk back"
            )
        )
    }
    
    private func createGenericShoppingSuggestion(weeklySpend: Double) -> ExplorationSuggestion {
        return ExplorationSuggestion(
            category: .shopping,
            newOption: RecommendedOption(
                name: "Diversified Shopping Strategy",
                description: "Spread errands across multiple optimized stops",
                location: "Planned multi-stop route",
                estimatedCost: weeklySpend,
                similarityToKnownPreferences: 0.8,
                qualityIndicators: [.fresh_ingredients, .local_institution]
            ),
            fallbackOption: RecommendedOption(
                name: "Single-store shopping",
                description: "Your current one-stop approach",
                location: "Usual supermarket",
                estimatedCost: weeklySpend,
                similarityToKnownPreferences: 0.7,
                qualityIndicators: [.similar_to_known_favorite]
            ),
            riskLevel: .low,
            confidence: 0.8,
            reasoning: "Distributed shopping often provides better quality and variety for the same cost",
            safetyNet: SafetyNet(
                exitStrategy: "Revert to single-store shopping anytime",
                timeCommitment: "May actually save time with better planning",
                costLimit: "Same weekly budget",
                fallbackDistance: "Original store always available"
            )
        )
    }
    
    /// Record exploration results to improve future suggestions
    public func recordExplorationResult(_ result: ExplorationResult) {
        explorationHistory.append(result)
        updateUserPattern(from: result)
    }
    
    private func updateUserPattern(from result: ExplorationResult) {
        // Learn from user's exploration outcomes to improve future suggestions
        if result.outcome == .loved_it || result.outcome == .liked_it {
            // Add successful explorations to user preferences
            var updatedPreferences = userPattern.knownPreferences
            updatedPreferences.append(result.suggestion.newOption.name)
            
            userPattern = UserPattern(
                knownPreferences: updatedPreferences,
                avoidances: userPattern.avoidances,
                comfortZone: userPattern.comfortZone,
                lastExplorations: userPattern.lastExplorations + [result],
                riskTolerance: userPattern.riskTolerance
            )
        }
    }
}

// MARK: - Default Patterns

extension UserPattern {
    static let defaultPattern = UserPattern(
        knownPreferences: ["Five Guys burgers", "Greek restaurant usual order"],
        avoidances: ["Spicy food", "Unfamiliar ethnic cuisines"],
        comfortZone: ["Nuremberg City Center", "Chain restaurants", "Tourist areas"],
        riskTolerance: .low
    )
    
    /// Pattern for users who want to explore but are risk-averse
    static let cautiousExplorer = UserPattern(
        knownPreferences: [],
        avoidances: [],
        comfortZone: [],
        riskTolerance: .minimal
    )
    
    /// Pattern for users ready to try more adventurous options
    static let adventurousExplorer = UserPattern(
        knownPreferences: [],
        avoidances: [],
        comfortZone: [],
        riskTolerance: .moderate
    )
} 