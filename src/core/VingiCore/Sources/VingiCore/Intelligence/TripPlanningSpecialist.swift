import Foundation

/// Specialized component for comprehensive trip planning that prevents common oversights
/// Specifically designed to avoid disasters like "arrived hungry with all restaurants closed"
public class TripPlanningSpecialist {
    
    // MARK: - Types
    
    public struct TripPlan: Codable {
        public let destination: String
        public let date: Date
        public let groupSize: Int
        public let essentialChecklist: [EssentialItem]
        public let contingencyPlans: [ContingencyPlan]
        public let timelineRecommendations: [TimelineItem]
        public let localIntelligence: LocalIntelligence
        
        public init(
            destination: String,
            date: Date,
            groupSize: Int,
            essentialChecklist: [EssentialItem],
            contingencyPlans: [ContingencyPlan],
            timelineRecommendations: [TimelineItem],
            localIntelligence: LocalIntelligence
        ) {
            self.destination = destination
            self.date = date
            self.groupSize = groupSize
            self.essentialChecklist = essentialChecklist
            self.contingencyPlans = contingencyPlans
            self.timelineRecommendations = timelineRecommendations
            self.localIntelligence = localIntelligence
        }
    }
    
    public struct EssentialItem: Identifiable, Codable {
        public let id: UUID
        public let category: EssentialCategory
        public let title: String
        public let description: String
        public let priority: Priority
        public let estimatedTime: TimeInterval
        public let canBeAutomated: Bool
        
        public init(
            id: UUID = UUID(),
            category: EssentialCategory,
            title: String,
            description: String,
            priority: Priority,
            estimatedTime: TimeInterval,
            canBeAutomated: Bool = false
        ) {
            self.id = id
            self.category = category
            self.title = title
            self.description = description
            self.priority = priority
            self.estimatedTime = estimatedTime
            self.canBeAutomated = canBeAutomated
        }
    }
    
    public enum EssentialCategory: String, CaseIterable, Codable {
        case food = "food"                    // The most overlooked but critical
        case transportation = "transportation"
        case accommodation = "accommodation"
        case attractions = "attractions"
        case emergencies = "emergencies"
        case communication = "communication"
        case weather = "weather"
        case money = "money"
    }
    
    public enum Priority: String, CaseIterable, Codable {
        case critical = "critical"    // Trip fails without this
        case important = "important"  // Significantly impacts experience
        case helpful = "helpful"      // Nice to have
    }
    
    public struct ContingencyPlan: Identifiable, Codable {
        public let id: UUID
        public let scenario: String
        public let backupOptions: [String]
        public let preventionSteps: [String]
        
        public init(
            id: UUID = UUID(),
            scenario: String,
            backupOptions: [String],
            preventionSteps: [String]
        ) {
            self.id = id
            self.scenario = scenario
            self.backupOptions = backupOptions
            self.preventionSteps = preventionSteps
        }
    }
    
    public struct TimelineItem: Identifiable, Codable {
        public let id: UUID
        public let timeframe: String
        public let tasks: [String]
        public let reasoning: String
        
        public init(
            id: UUID = UUID(),
            timeframe: String,
            tasks: [String],
            reasoning: String
        ) {
            self.id = id
            self.timeframe = timeframe
            self.tasks = tasks
            self.reasoning = reasoning
        }
    }
    
    public struct LocalIntelligence: Codable {
        public let restaurantHours: RestaurantInfo
        public let attractionHours: [String: String]
        public let transportationNotes: [String]
        public let localCustoms: [String]
        public let emergencyContacts: [String]
        
        public init(
            restaurantHours: RestaurantInfo,
            attractionHours: [String: String],
            transportationNotes: [String],
            localCustoms: [String],
            emergencyContacts: [String]
        ) {
            self.restaurantHours = restaurantHours
            self.attractionHours = attractionHours
            self.transportationNotes = transportationNotes
            self.localCustoms = localCustoms
            self.emergencyContacts = emergencyContacts
        }
    }
    
    public struct RestaurantInfo: Codable {
        public let typicalOpeningHours: String
        public let lunchBreaks: String?
        public let dayOfWeekVariations: [String: String]
        public let seasonalChanges: String?
        public let recommendedOptions: [String]
        public let backupOptions: [String]
        
        public init(
            typicalOpeningHours: String,
            lunchBreaks: String? = nil,
            dayOfWeekVariations: [String: String] = [:],
            seasonalChanges: String? = nil,
            recommendedOptions: [String] = [],
            backupOptions: [String] = []
        ) {
            self.typicalOpeningHours = typicalOpeningHours
            self.lunchBreaks = lunchBreaks
            self.dayOfWeekVariations = dayOfWeekVariations
            self.seasonalChanges = seasonalChanges
            self.recommendedOptions = recommendedOptions
            self.backupOptions = backupOptions
        }
    }
    
    // MARK: - Public Methods
    
    /// Create comprehensive trip plan for Ansbach-style destinations
    public static func planTripToSmallGermanTown(
        destination: String,
        date: Date,
        groupSize: Int = 1
    ) -> TripPlan {
        
        let essentials = createEssentialChecklist(for: destination, groupSize: groupSize, date: date)
        let contingencies = createContingencyPlans(for: destination)
        let timeline = createTimeline(for: destination, date: date)
        let intelligence = gatherLocalIntelligence(for: destination)
        
        return TripPlan(
            destination: destination,
            date: date,
            groupSize: groupSize,
            essentialChecklist: essentials,
            contingencyPlans: contingencies,
            timelineRecommendations: timeline,
            localIntelligence: intelligence
        )
    }
    
    /// Specific plan for Ansbach based on your disaster experience
    public static func planAnsbachTrip(date: Date, groupSize: Int = 2) -> TripPlan {
        
        let essentials: [EssentialItem] = [
            // FOOD FIRST - This is what went wrong!
            EssentialItem(
                category: .food,
                title: "Restaurant Research & Reservations",
                description: "Find 3-4 restaurants in Ansbach, check opening hours for your specific day, make reservations if possible",
                priority: .critical,
                estimatedTime: 900, // 15 minutes
                canBeAutomated: true
            ),
            EssentialItem(
                category: .food,
                title: "Backup Food Plan",
                description: "Identify supermarkets, bakeries, food trucks. Pack snacks and water. Know where to get emergency food.",
                priority: .critical,
                estimatedTime: 300, // 5 minutes
                canBeAutomated: false
            ),
            EssentialItem(
                category: .food,
                title: "Check Sunday/Holiday Hours",
                description: "German restaurants often close Sundays or have unusual hours. Verify for your exact visit date.",
                priority: .critical,
                estimatedTime: 180, // 3 minutes
                canBeAutomated: true
            ),
            
            // Other essentials
            EssentialItem(
                category: .attractions,
                title: "Walled City Opening Hours",
                description: "Verify the historic center and main attractions are accessible on your visit day",
                priority: .important,
                estimatedTime: 300,
                canBeAutomated: true
            ),
            EssentialItem(
                category: .transportation,
                title: "Transportation Plan",
                description: "Book train tickets or plan driving route. Check for construction/delays.",
                priority: .important,
                estimatedTime: 600,
                canBeAutomated: false
            ),
            EssentialItem(
                category: .money,
                title: "Cash and Payment Options",
                description: "Many small German town establishments are cash-only. Bring euros and find ATMs.",
                priority: .important,
                estimatedTime: 120,
                canBeAutomated: false
            )
        ]
        
        let contingencies: [ContingencyPlan] = [
            ContingencyPlan(
                scenario: "All restaurants are closed (like what happened to you)",
                backupOptions: [
                    "Rewe or Edeka supermarket for basics",
                    "Bakeries usually open until noon on Sundays",
                    "Gas stations with food sections",
                    "Drive to nearby larger town (Rothenburg is 30min away)",
                    "Packed snacks and water from home"
                ],
                preventionSteps: [
                    "Call restaurants day before to confirm hours",
                    "Make reservations where possible",
                    "Always pack emergency snacks",
                    "Check if visit day is a German holiday"
                ]
            ),
            ContingencyPlan(
                scenario: "Walled city/attractions closed unexpectedly",
                backupOptions: [
                    "Walk around the historic center (always accessible)",
                    "Visit nearby Rothenburg ob der Tauber",
                    "Explore local parks and nature areas",
                    "Find local beer gardens or cafes"
                ],
                preventionSteps: [
                    "Check tourist information website",
                    "Call ahead if possible",
                    "Have backup destinations in mind"
                ]
            ),
            ContingencyPlan(
                scenario: "Transportation delays or cancellations",
                backupOptions: [
                    "Alternative train routes via Nuremberg",
                    "Bus connections as backup",
                    "Rideshare or taxi for group",
                    "Adjust timing or reschedule if necessary"
                ],
                preventionSteps: [
                    "Book flexible tickets when possible",
                    "Check DB app for real-time updates",
                    "Leave extra time for connections",
                    "Have contact info for group coordination"
                ]
            )
        ]
        
        let timeline: [TimelineItem] = [
            TimelineItem(
                timeframe: "1 week before",
                tasks: [
                    "Research restaurants and make reservations",
                    "Check all opening hours for your specific date",
                    "Book transportation tickets",
                    "Coordinate with group on meeting times"
                ],
                reasoning: "Restaurants in small towns book up fast, especially on weekends"
            ),
            TimelineItem(
                timeframe: "2-3 days before",
                tasks: [
                    "Confirm restaurant reservations",
                    "Check weather forecast and pack accordingly",
                    "Withdraw cash from ATM",
                    "Pack emergency snacks and water"
                ],
                reasoning: "Last-minute confirmations prevent surprises"
            ),
            TimelineItem(
                timeframe: "Day before",
                tasks: [
                    "Final check of train times",
                    "Call restaurants to reconfirm (if no reservations)",
                    "Pack day bag with snacks, water, cash",
                    "Share final itinerary with group"
                ],
                reasoning: "Day-of changes are common in small towns"
            ),
            TimelineItem(
                timeframe: "Day of travel",
                tasks: [
                    "Eat proper breakfast before leaving",
                    "Bring extra snacks 'just in case'",
                    "Have restaurant phone numbers saved",
                    "Start with lunch plan, then sightseeing"
                ],
                reasoning: "Food security first, then enjoy the sights"
            )
        ]
        
        let intelligence = LocalIntelligence(
            restaurantHours: RestaurantInfo(
                typicalOpeningHours: "11:30-14:00, 17:30-22:00",
                lunchBreaks: "Many close 14:00-17:30",
                dayOfWeekVariations: [
                    "Sunday": "Many closed or limited hours",
                    "Monday": "Some restaurants closed (Ruhetag)",
                    "Saturday": "Usually normal hours"
                ],
                seasonalChanges: "Winter hours may be shorter",
                recommendedOptions: [
                    "Gasthof zum Goldenen Hirsch",
                    "Restaurant Bernardino",
                    "Brauereigasthof Bärenturm"
                ],
                backupOptions: [
                    "Rewe supermarket (usually open until 20:00)",
                    "Bäckerei Seifert (traditional German bakery)",
                    "Döner shops (often have longer hours)"
                ]
            ),
            attractionHours: [
                "Residenz Ansbach": "9:00-18:00 (closed Mondays)",
                "St. Gumbertus Church": "Usually open during day",
                "Historic Old Town": "Always accessible for walking"
            ],
            transportationNotes: [
                "Direct train from Nuremberg takes ~45 minutes",
                "Ansbach station is walkable to old town (~10 minutes)",
                "Free parking available at Residenz if driving",
                "Last trains back to major cities around 22:00"
            ],
            localCustoms: [
                "Greet shopkeepers with 'Guten Tag'",
                "Many places are cash-only",
                "Lunch is typically 12:00-14:00",
                "Quiet time (Ruhezeit) observed 13:00-15:00 and after 22:00"
            ],
            emergencyContacts: [
                "Tourist Information: +49 981 51-243",
                "Emergency: 112",
                "Local Police: +49 981 9094-0"
            ]
        )
        
        return TripPlan(
            destination: "Ansbach",
            date: date,
            groupSize: groupSize,
            essentialChecklist: essentials,
            contingencyPlans: contingencies,
            timelineRecommendations: timeline,
            localIntelligence: intelligence
        )
    }
    
    // MARK: - Private Helper Methods
    
    private static func createEssentialChecklist(
        for destination: String,
        groupSize: Int,
        date: Date
    ) -> [EssentialItem] {
        // Default checklist for any small German town
        return [
            EssentialItem(
                category: .food,
                title: "Restaurant Research",
                description: "Find and verify restaurants open on your visit day",
                priority: .critical,
                estimatedTime: 600,
                canBeAutomated: true
            ),
            EssentialItem(
                category: .food,
                title: "Emergency Food Plan",
                description: "Pack snacks and identify backup food sources",
                priority: .critical,
                estimatedTime: 300,
                canBeAutomated: false
            ),
            EssentialItem(
                category: .transportation,
                title: "Travel Arrangements",
                description: "Book tickets and plan route",
                priority: .important,
                estimatedTime: 900,
                canBeAutomated: false
            )
        ]
    }
    
    private static func createContingencyPlans(for destination: String) -> [ContingencyPlan] {
        return [
            ContingencyPlan(
                scenario: "No restaurants open",
                backupOptions: ["Supermarket", "Bakery", "Nearby town", "Packed food"],
                preventionSteps: ["Verify hours in advance", "Make reservations", "Pack snacks"]
            )
        ]
    }
    
    private static func createTimeline(for destination: String, date: Date) -> [TimelineItem] {
        return [
            TimelineItem(
                timeframe: "1 week before",
                tasks: ["Research restaurants", "Book transportation"],
                reasoning: "Advance planning prevents disasters"
            )
        ]
    }
    
    private static func gatherLocalIntelligence(for destination: String) -> LocalIntelligence {
        return LocalIntelligence(
            restaurantHours: RestaurantInfo(typicalOpeningHours: "11:30-22:00"),
            attractionHours: [:],
            transportationNotes: [],
            localCustoms: [],
            emergencyContacts: []
        )
    }
} 