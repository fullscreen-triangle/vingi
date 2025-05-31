import XCTest
@testable import VingiCore

final class CognitivePatternTests: XCTestCase {
    
    var patternDetector: CognitivePatternDetector!
    var selectiveAbilityEngine: SelectiveAbilityRecognitionEngine!
    
    override func setUp() {
        super.setUp()
        patternDetector = CognitivePatternDetector()
        selectiveAbilityEngine = SelectiveAbilityRecognitionEngine()
    }
    
    override func tearDown() {
        patternDetector = nil
        selectiveAbilityEngine = nil
        super.tearDown()
    }
    
    // MARK: - Analysis Paralysis Tests
    
    func testAnalysisParalysisDetection() async throws {
        // Create sample decision events that indicate paralysis
        let decisions = [
            DecisionEvent(
                domain: "Transportation",
                researchTime: 2400, // 40 minutes - high research time
                optionsConsidered: 15, // Many options considered
                decisionMade: false // No decision made
            ),
            DecisionEvent(
                domain: "Transportation", 
                researchTime: 1800, // 30 minutes
                optionsConsidered: 12,
                decisionMade: false
            )
        ]
        
        let behaviorData = UserBehaviorData(
            recentDecisions: decisions,
            planningActivity: [],
            routineChoices: [],
            performanceHistory: []
        )
        
        let patterns = await patternDetector.detectActivePatterns(userBehavior: behaviorData)
        
        // Should detect analysis paralysis
        let paralysisPattern = patterns.first { pattern in
            if case .analysisParalysis = pattern {
                return true
            }
            return false
        }
        
        XCTAssertNotNil(paralysisPattern, "Should detect analysis paralysis pattern")
        
        if case let .analysisParalysis(severity, domain, interventions) = paralysisPattern! {
            XCTAssertEqual(domain, "Transportation")
            XCTAssertFalse(interventions.isEmpty, "Should provide interventions")
            XCTAssertTrue(severity == .moderate || severity == .severe, "Should detect moderate to severe paralysis")
        }
    }
    
    // MARK: - Tunnel Vision Tests
    
    func testTunnelVisionDetection() async throws {
        // Create planning events that show tunnel vision
        let planningEvents = [
            PlanningEvent(
                activity: "Transportation research",
                domain: "Transportation",
                timeSpent: 7200, // 2 hours on transportation
                detailLevel: 0.9,
                completeness: 0.9
            ),
            PlanningEvent(
                activity: "Route optimization",
                domain: "Transportation", 
                timeSpent: 3600, // 1 hour more on transportation
                detailLevel: 0.8,
                completeness: 0.8
            ),
            PlanningEvent(
                activity: "Food planning",
                domain: "Food",
                timeSpent: 300, // Only 5 minutes on food
                detailLevel: 0.1,
                completeness: 0.1
            )
        ]
        
        let behaviorData = UserBehaviorData(
            recentDecisions: [],
            planningActivity: planningEvents,
            routineChoices: [],
            performanceHistory: []
        )
        
        let patterns = await patternDetector.detectActivePatterns(userBehavior: behaviorData)
        
        // Should detect tunnel vision
        let tunnelPattern = patterns.first { pattern in
            if case .tunnelVision = pattern {
                return true
            }
            return false
        }
        
        XCTAssertNotNil(tunnelPattern, "Should detect tunnel vision pattern")
        
        if case let .tunnelVision(focusDomain, neglectedDomains, riskLevel) = tunnelPattern! {
            XCTAssertEqual(focusDomain, "Transportation")
            XCTAssertTrue(neglectedDomains.contains("Food"))
            XCTAssertTrue(riskLevel == .high || riskLevel == .critical)
        }
    }
    
    // MARK: - Default Loop Tests
    
    func testDefaultLoopDetection() async throws {
        // Create choice events that show repetitive behavior
        let choices = [
            ChoiceEvent(
                domain: "Dining",
                choice: "Five Guys",
                alternatives: ["McDonald's", "Subway", "Local restaurant"],
                satisfaction: 0.7,
                metadata: ["location": "City Center"]
            ),
            ChoiceEvent(
                domain: "Dining",
                choice: "Five Guys",
                alternatives: ["McDonald's", "Burger King"],
                satisfaction: 0.6,
                metadata: ["location": "City Center"]
            ),
            ChoiceEvent(
                domain: "Dining",
                choice: "Five Guys",
                alternatives: ["Subway"],
                satisfaction: 0.5,
                metadata: ["location": "City Center"]
            ),
            ChoiceEvent(
                domain: "Dining",
                choice: "Five Guys",
                alternatives: ["McDonald's"],
                satisfaction: 0.6,
                metadata: ["location": "City Center"]
            ),
            ChoiceEvent(
                domain: "Dining",
                choice: "Five Guys",
                alternatives: ["Burger King"],
                satisfaction: 0.7,
                metadata: ["location": "City Center"]
            )
        ]
        
        let behaviorData = UserBehaviorData(
            recentDecisions: [],
            planningActivity: [],
            routineChoices: choices,
            performanceHistory: []
        )
        
        let patterns = await patternDetector.detectActivePatterns(userBehavior: behaviorData)
        
        // Should detect default loop
        let loopPattern = patterns.first { pattern in
            if case .defaultLoop = pattern {
                return true
            }
            return false
        }
        
        XCTAssertNotNil(loopPattern, "Should detect default loop pattern")
        
        if case let .defaultLoop(domain, constraintType, optimizationPotential) = loopPattern! {
            XCTAssertEqual(domain, "Dining")
            XCTAssertTrue(optimizationPotential > 0, "Should have optimization potential")
        }
    }
    
    // MARK: - Exceptional Ability Self-Doubt Tests
    
    func testExceptionalAbilitySelfDoubtDetection() async throws {
        // Create performance events that show high performance but low confidence
        let performanceEvents = [
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .meaningfulSequences,
                context: .highStakes,
                accuracy: 0.85, // High actual performance
                confidence: 0.25, // Low confidence
                description: "Bank number recall"
            ),
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .meaningfulSequences,
                context: .timePressed,
                accuracy: 0.9, // Very high performance
                confidence: 0.3, // Low confidence
                description: "Account details recall"
            ),
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .meaningfulSequences,
                context: .highStakes,
                accuracy: 0.8, // High performance
                confidence: 0.2, // Very low confidence
                description: "Important sequence recall"
            )
        ]
        
        let behaviorData = UserBehaviorData(
            recentDecisions: [],
            planningActivity: [],
            routineChoices: [],
            performanceHistory: performanceEvents
        )
        
        let patterns = await patternDetector.detectActivePatterns(userBehavior: behaviorData)
        
        // Should detect exceptional ability self-doubt
        let doubtPattern = patterns.first { pattern in
            if case .exceptionalAbilitySelfDoubt = pattern {
                return true
            }
            return false
        }
        
        XCTAssertNotNil(doubtPattern, "Should detect exceptional ability self-doubt pattern")
        
        if case let .exceptionalAbilitySelfDoubt(ability, evidenceCount, actualPerformance, perceivedPerformance, confidenceGap) = doubtPattern! {
            XCTAssertEqual(ability, .meaningfulSequences)
            XCTAssertEqual(evidenceCount, 3)
            XCTAssertGreaterThan(actualPerformance, 0.7, "Actual performance should be high")
            XCTAssertLessThan(perceivedPerformance, 0.5, "Perceived performance should be low")
            XCTAssertGreaterThan(confidenceGap, 0.3, "Confidence gap should be significant")
        }
    }
    
    // MARK: - Selective Ability Engine Tests
    
    func testAbilityMapping() async throws {
        let performanceEvents = [
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .emotionalMemories,
                context: .emotionallyCharged,
                accuracy: 1.0,
                confidence: 0.95,
                description: "First words memory"
            ),
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .routineInformation,
                context: .routineTask,
                accuracy: 0.2,
                confidence: 0.1,
                description: "Password recall"
            )
        ]
        
        let abilityMap = await selectiveAbilityEngine.mapAbilityDomains(performanceHistory: performanceEvents)
        
        XCTAssertNotNil(abilityMap)
        XCTAssertFalse(abilityMap.domains.isEmpty, "Should map at least some domains")
        XCTAssertGreaterThan(abilityMap.overallConfidence, 0, "Should have positive overall confidence")
        
        // Check emotional memories domain should have high performance
        if let emotionalProfile = abilityMap.domains[.emotionalMemories] {
            XCTAssertGreaterThan(emotionalProfile.averageAccuracy, 0.9, "Emotional memories should have high accuracy")
        }
        
        // Check routine information should have low performance
        if let routineProfile = abilityMap.domains[.routineInformation] {
            XCTAssertLessThan(routineProfile.averageAccuracy, 0.5, "Routine information should have low accuracy")
        }
    }
    
    func testOptimalConditionsIdentification() async throws {
        // Add some performance events to the engine first
        let events = [
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .meaningfulSequences,
                context: .highStakes,
                accuracy: 0.9,
                confidence: 0.4,
                description: "High stakes recall"
            ),
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .meaningfulSequences,
                context: .highStakes,
                accuracy: 0.85,
                confidence: 0.3,
                description: "Another high stakes recall"
            ),
            SelectiveAbilityRecognitionEngine.PerformanceEvent(
                domain: .meaningfulSequences,
                context: .routineTask,
                accuracy: 0.3,
                confidence: 0.2,
                description: "Routine recall"
            )
        ]
        
        // Record events
        for event in events {
            selectiveAbilityEngine.recordPerformanceEvent(event)
        }
        
        let optimalConditions = await selectiveAbilityEngine.identifyOptimalConditions(ability: .meaningfulSequences)
        
        XCTAssertFalse(optimalConditions.isEmpty, "Should identify optimal conditions")
        
        // High stakes should be the optimal condition
        let highStakesCondition = optimalConditions.first { $0.contextType == .highStakes }
        XCTAssertNotNil(highStakesCondition, "Should identify high stakes as optimal")
        XCTAssertGreaterThan(highStakesCondition!.activationProbability, 0.5, "High stakes should have high activation probability")
    }
    
    func testConfidenceStrategyGeneration() async throws {
        let strategy = await selectiveAbilityEngine.buildContextualConfidence(
            domain: .meaningfulSequences,
            context: .highStakes
        )
        
        XCTAssertFalse(strategy.implementationSteps.isEmpty, "Should provide implementation steps")
        XCTAssertFalse(strategy.description.isEmpty, "Should provide strategy description")
        XCTAssertFalse(strategy.expectedOutcome.isEmpty, "Should provide expected outcome")
    }
    
    // MARK: - Cognitive Load Assessment Tests
    
    func testCognitiveLoadAssessment() async throws {
        let userContext = UserContext(
            currentTask: "Complex planning task",
            recentDecisionCount: 8,
            contextSwitchCount: 5,
            researchTime: 3600, // 1 hour
            planningActivity: [],
            routineChoices: [],
            performanceHistory: []
        )
        
        let assessment = await patternDetector.assessCognitiveLoad(userContext: userContext)
        
        XCTAssertGreaterThan(assessment.overallLoad, 0, "Should assess some cognitive load")
        XCTAssertFalse(assessment.patternRisks.isEmpty, "Should assess pattern risks")
        XCTAssertNotEqual(assessment.urgencyLevel, .monitoring, "Should indicate attention or higher urgency")
    }
    
    // MARK: - Integration Tests
    
    func testFullPatternDetectionWorkflow() async throws {
        // Create comprehensive behavior data that triggers multiple patterns
        let behaviorData = UserBehaviorData(
            recentDecisions: [
                DecisionEvent(
                    domain: "Travel",
                    researchTime: 2100,
                    optionsConsidered: 10,
                    decisionMade: false
                )
            ],
            planningActivity: [
                PlanningEvent(
                    activity: "Transportation planning",
                    domain: "Transportation",
                    timeSpent: 5400, // 1.5 hours
                    detailLevel: 0.9,
                    completeness: 0.8
                ),
                PlanningEvent(
                    activity: "Food planning",
                    domain: "Food",
                    timeSpent: 180, // 3 minutes
                    detailLevel: 0.1,
                    completeness: 0.1
                )
            ],
            routineChoices: [
                ChoiceEvent(domain: "Shopping", choice: "Same store", alternatives: ["Other stores"]),
                ChoiceEvent(domain: "Shopping", choice: "Same store", alternatives: ["Other stores"]),
                ChoiceEvent(domain: "Shopping", choice: "Same store", alternatives: ["Other stores"]),
                ChoiceEvent(domain: "Shopping", choice: "Same store", alternatives: ["Other stores"]),
                ChoiceEvent(domain: "Shopping", choice: "Same store", alternatives: ["Other stores"])
            ],
            performanceHistory: [
                SelectiveAbilityRecognitionEngine.PerformanceEvent(
                    domain: .meaningfulSequences,
                    context: .highStakes,
                    accuracy: 0.8,
                    confidence: 0.2,
                    description: "Important recall"
                )
            ]
        )
        
        let patterns = await patternDetector.detectActivePatterns(userBehavior: behaviorData)
        
        // Should detect multiple patterns
        XCTAssertGreaterThan(patterns.count, 1, "Should detect multiple cognitive patterns")
        
        // Verify we can detect different types
        let patternTypes = patterns.map { pattern in
            switch pattern {
            case .analysisParalysis: return "paralysis"
            case .tunnelVision: return "tunnel"
            case .defaultLoop: return "loop"
            case .exceptionalAbilitySelfDoubt: return "doubt"
            }
        }
        
        XCTAssertTrue(Set(patternTypes).count > 1, "Should detect patterns of different types")
    }
} 