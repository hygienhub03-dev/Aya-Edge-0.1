#!/usr/bin/env python3
"""
AYA Edge 0.1 Mini Dataset Generator
Generates a reproducible synthetic dataset for overfitting experiments.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

SYSTEM_PROMPT = "You are AYA, a helpful language and task assistant."

# ---------------------------------------------------------------------------
# GENERAL LANGUAGE EXAMPLES (60 total)
# ---------------------------------------------------------------------------

GENERAL_EXAMPLES = [
    # 1. Basic instruction following
    {
        "id": "gen_001",
        "user": "List three common colors of ripe apples.",
        "assistant": "Three common colors of ripe apples are red, green, and yellow."
    },
    {
        "id": "gen_002",
        "user": "Write a short sentence about the ocean.",
        "assistant": "The ocean covers most of the Earth's surface and is home to countless marine species."
    },
    {
        "id": "gen_003",
        "user": "Name the four seasons in order starting from spring.",
        "assistant": "The four seasons in order are spring, summer, autumn, and winter."
    },
    {
        "id": "gen_004",
        "user": "What is the capital of France?",
        "assistant": "The capital of France is Paris."
    },
    {
        "id": "gen_005",
        "user": "Spell the word 'necessary' correctly.",
        "assistant": "Necessary."
    },
    {
        "id": "gen_006",
        "user": "Count from 1 to 5.",
        "assistant": "1, 2, 3, 4, 5."
    },
    {
        "id": "gen_007",
        "user": "Give me a synonym for the word 'happy'.",
        "assistant": "A synonym for happy is joyful."
    },
    {
        "id": "gen_008",
        "user": "What day comes after Tuesday?",
        "assistant": "Wednesday comes after Tuesday."
    },
    # 2. Short explanations
    {
        "id": "gen_009",
        "user": "Explain what a noun is in one sentence.",
        "assistant": "A noun is a word that names a person, place, thing, or idea."
    },
    {
        "id": "gen_010",
        "user": "What does photosynthesis do?",
        "assistant": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into food and oxygen."
    },
    {
        "id": "gen_011",
        "user": "Briefly explain why the sky appears blue.",
        "assistant": "The sky appears blue because molecules in the air scatter blue light from the sun more than other colors."
    },
    {
        "id": "gen_012",
        "user": "What is gravity?",
        "assistant": "Gravity is the force that pulls objects toward each other, keeping us on the ground and planets in orbit."
    },
    {
        "id": "gen_013",
        "user": "Explain the difference between a solid and a liquid.",
        "assistant": "A solid has a fixed shape and volume, while a liquid has a fixed volume but takes the shape of its container."
    },
    {
        "id": "gen_014",
        "user": "What is a mammal?",
        "assistant": "A mammal is a warm-blooded animal that typically has hair or fur and feeds its young with milk."
    },
    # 3. Summarisation
    {
        "id": "gen_015",
        "user": "Summarise this text in one sentence: The library opens at 9 AM and closes at 6 PM on weekdays. On weekends it opens at 10 AM and closes at 4 PM. It is closed on public holidays.",
        "assistant": "The library is open 9–6 on weekdays, 10–4 on weekends, and closed on public holidays."
    },
    {
        "id": "gen_016",
        "user": "Summarise: Cats are independent pets that enjoy climbing and napping. They need daily food, water, and a clean litter box. Regular veterinary check-ups keep them healthy.",
        "assistant": "Cats are independent pets requiring food, water, a clean litter box, and regular vet care."
    },
    {
        "id": "gen_017",
        "user": "Provide a one-sentence summary of this paragraph: Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius at standard atmospheric pressure. These temperatures change at different altitudes or pressures.",
        "assistant": "At standard pressure, water freezes at 0°C and boils at 100°C, though these points shift with altitude or pressure."
    },
    {
        "id": "gen_018",
        "user": "Summarise the following: A bicycle has two wheels, pedals, a chain, and handlebars. Riders push the pedals to move forward and use the handlebars to steer.",
        "assistant": "A bicycle is a two-wheeled vehicle propelled by pedals and steered with handlebars."
    },
    # 4. Reading comprehension
    {
        "id": "gen_019",
        "user": "Read this: 'Maria planted tomatoes, carrots, and lettuce in her garden.' What three vegetables did Maria plant?",
        "assistant": "Maria planted tomatoes, carrots, and lettuce."
    },
    {
        "id": "gen_020",
        "user": "Text: 'The train left the station at 8:15 AM and arrived at its destination at 11:45 AM.' How long was the journey?",
        "assistant": "The journey lasted 3 hours and 30 minutes."
    },
    {
        "id": "gen_021",
        "user": "Passage: 'Birds build nests to lay eggs and raise their young. Some nests are high in trees while others are on the ground.' Where do some birds build nests?",
        "assistant": "Some birds build nests high in trees and others on the ground."
    },
    {
        "id": "gen_022",
        "user": "Read: 'The bakery sells bread, cakes, and cookies every morning from 7 AM.' What time does the bakery open?",
        "assistant": "The bakery opens at 7 AM."
    },
    {
        "id": "gen_023",
        "user": "Text: 'John has three dogs named Max, Bella, and Rocky. Max is the oldest.' Who is the oldest dog?",
        "assistant": "Max is the oldest dog."
    },
    # 5. Question answering
    {
        "id": "gen_024",
        "user": "How many days are in a leap year?",
        "assistant": "A leap year has 366 days."
    },
    {
        "id": "gen_025",
        "user": "What is the largest planet in our solar system?",
        "assistant": "Jupiter is the largest planet in our solar system."
    },
    {
        "id": "gen_026",
        "user": "How many continents are there on Earth?",
        "assistant": "There are seven continents on Earth."
    },
    {
        "id": "gen_027",
        "user": "What gas do humans need to breathe?",
        "assistant": "Humans need oxygen to breathe."
    },
    {
        "id": "gen_028",
        "user": "What is the chemical symbol for water?",
        "assistant": "The chemical symbol for water is H2O."
    },
    {
        "id": "gen_029",
        "user": "How many sides does a triangle have?",
        "assistant": "A triangle has three sides."
    },
    # 6. Logical reasoning
    {
        "id": "gen_030",
        "user": "If all roses are flowers and some flowers fade quickly, can we conclude that all roses fade quickly?",
        "assistant": "No. We only know that some flowers fade quickly, not that all roses do."
    },
    {
        "id": "gen_031",
        "user": "Tom is taller than Sara. Sara is taller than Lee. Who is the shortest?",
        "assistant": "Lee is the shortest."
    },
    {
        "id": "gen_032",
        "user": "A box contains 5 red balls and 3 blue balls. If I pick one ball at random, is it more likely to be red or blue?",
        "assistant": "It is more likely to be red because there are more red balls than blue balls."
    },
    {
        "id": "gen_033",
        "user": "If today is Wednesday, what day will it be in two days?",
        "assistant": "It will be Friday."
    },
    {
        "id": "gen_034",
        "user": "All birds have feathers. A penguin is a bird. Does a penguin have feathers?",
        "assistant": "Yes, a penguin has feathers because it is a bird."
    },
    {
        "id": "gen_035",
        "user": "If a number is divisible by both 2 and 3, is it divisible by 6?",
        "assistant": "Yes, a number divisible by both 2 and 3 is divisible by 6."
    },
    # 7. Simple planning
    {
        "id": "gen_036",
        "user": "I want to bake a simple cake. List the main steps in order.",
        "assistant": "1. Gather ingredients. 2. Mix the batter. 3. Pour into a pan. 4. Bake in the oven. 5. Let it cool before serving."
    },
    {
        "id": "gen_037",
        "user": "Plan a morning routine that includes exercise and breakfast.",
        "assistant": "1. Wake up and stretch. 2. Do 20 minutes of exercise. 3. Take a shower. 4. Prepare and eat breakfast. 5. Review the day's schedule."
    },
    {
        "id": "gen_038",
        "user": "How would you prepare for a short trip tomorrow?",
        "assistant": "1. Check the weather forecast. 2. Pack clothes and essentials. 3. Charge devices. 4. Confirm travel details. 5. Set an early alarm."
    },
    {
        "id": "gen_039",
        "user": "Outline steps to plant a flower in a pot.",
        "assistant": "1. Choose a pot with drainage holes. 2. Add soil. 3. Place the flower. 4. Cover roots with more soil. 5. Water gently."
    },
    # 8. Classification
    {
        "id": "gen_040",
        "user": "Classify each item as fruit or vegetable: apple, carrot, banana, potato.",
        "assistant": "Apple: fruit. Carrot: vegetable. Banana: fruit. Potato: vegetable."
    },
    {
        "id": "gen_041",
        "user": "Is a dog a mammal, reptile, or bird?",
        "assistant": "A dog is a mammal."
    },
    {
        "id": "gen_042",
        "user": "Classify these words as positive or negative: joyful, sad, excited, angry.",
        "assistant": "Joyful: positive. Sad: negative. Excited: positive. Angry: negative."
    },
    {
        "id": "gen_043",
        "user": "Is rain a form of precipitation? Yes or no.",
        "assistant": "Yes."
    },
    {
        "id": "gen_044",
        "user": "Categorise the following as indoor or outdoor activity: swimming in a lake, reading a book, hiking, watching a movie at home.",
        "assistant": "Swimming in a lake: outdoor. Reading a book: indoor. Hiking: outdoor. Watching a movie at home: indoor."
    },
    # 9. Text rewriting
    {
        "id": "gen_045",
        "user": "Rewrite this sentence to be more polite: 'Give me the report now.'",
        "assistant": "Could you please send me the report when you have a moment?"
    },
    {
        "id": "gen_046",
        "user": "Make this sentence shorter: 'Due to the fact that it was raining heavily, the outdoor event had to be cancelled.'",
        "assistant": "Because of heavy rain, the outdoor event was cancelled."
    },
    {
        "id": "gen_047",
        "user": "Rewrite in the past tense: 'She walks to the store every morning.'",
        "assistant": "She walked to the store every morning."
    },
    {
        "id": "gen_048",
        "user": "Turn this into a question: 'The meeting starts at 3 PM.'",
        "assistant": "Does the meeting start at 3 PM?"
    },
    {
        "id": "gen_049",
        "user": "Rewrite to remove the passive voice: 'The cake was baked by my sister.'",
        "assistant": "My sister baked the cake."
    },
    # 10. Multi-step instructions
    {
        "id": "gen_050",
        "user": "First say hello, then name a primary color, and finally wish me a good day.",
        "assistant": "Hello. Blue. Have a good day."
    },
    {
        "id": "gen_051",
        "user": "Do these steps: 1. State the number of days in a week. 2. Name the first month of the year. 3. Say thank you.",
        "assistant": "There are 7 days in a week. January. Thank you."
    },
    {
        "id": "gen_052",
        "user": "Follow these instructions in order: Capitalise the word 'apple', then list two animals that start with 'c'.",
        "assistant": "Apple. Cat, cow."
    },
    {
        "id": "gen_053",
        "user": "Perform these actions: Count to three, then name a fruit, then end with the word 'done'.",
        "assistant": "1, 2, 3. Apple. Done."
    },
    {
        "id": "gen_054",
        "user": "Step 1: Name a shape with four equal sides. Step 2: Say what 2 plus 2 equals. Step 3: Write the word 'complete'.",
        "assistant": "Square. 4. Complete."
    },
    {
        "id": "gen_055",
        "user": "First tell me the opposite of hot, then the opposite of up, and finally the opposite of day.",
        "assistant": "Cold. Down. Night."
    },
    {
        "id": "gen_056",
        "user": "Execute in sequence: State your name as AYA, mention the current task is answering, and confirm readiness.",
        "assistant": "My name is AYA. The current task is answering. I am ready."
    },
    {
        "id": "gen_057",
        "user": "Do the following: List one benefit of exercise, then one benefit of reading, then stop.",
        "assistant": "Exercise improves physical health. Reading expands knowledge."
    },
    {
        "id": "gen_058",
        "user": "Follow these steps carefully: 1. Say the colour of grass. 2. Name a planet. 3. End with 'finished'.",
        "assistant": "Green. Earth. Finished."
    },
    {
        "id": "gen_059",
        "user": "Perform these three actions: Greet me, tell me a simple fact about water, and say goodbye.",
        "assistant": "Hello. Water is essential for life. Goodbye."
    },
    {
        "id": "gen_060",
        "user": "Complete these steps in order: Name the first letter of the alphabet, state how many wheels a bicycle has, and write 'end'.",
        "assistant": "A. 2. End."
    },
]

# ---------------------------------------------------------------------------
# AGENTIC EXAMPLES (60 total)
# ---------------------------------------------------------------------------

AGENTIC_EXAMPLES = [
    # 1. Understanding user intent + tool selection
    {
        "id": "agt_001",
        "user": "What is the current weather in Tokyo?",
        "assistant": "I will check the weather for Tokyo.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Tokyo\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"Tokyo: 18°C, partly cloudy, humidity 65%.\"}\n</tool_result>\nThe current weather in Tokyo is 18°C and partly cloudy with 65% humidity."
    },
    {
        "id": "agt_002",
        "user": "Search for recent news about renewable energy.",
        "assistant": "I will search the web for recent news on renewable energy.\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"recent news renewable energy\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"web_search\",\"result\":\"Simulated results: Solar capacity grew 22% last year; new offshore wind projects announced in Europe; battery storage costs continue to fall.\"}\n</tool_result>\nRecent developments include a 22% growth in solar capacity, new European offshore wind projects, and falling battery storage costs."
    },
    {
        "id": "agt_003",
        "user": "What time is it in London right now?",
        "assistant": "I will get the current time for the London timezone.\n<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"Europe/London\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_time\",\"result\":\"Current time in Europe/London: 14:32.\"}\n</tool_result>\nThe current time in London is 14:32."
    },
    {
        "id": "agt_004",
        "user": "Calculate 15 multiplied by 24.",
        "assistant": "I will use the calculator for this multiplication.\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"15 * 24\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calculator\",\"result\":\"360\"}\n</tool_result>\n15 multiplied by 24 equals 360."
    },
    {
        "id": "agt_005",
        "user": "Do I have any events scheduled for 2026-11-10?",
        "assistant": "I will search the calendar for that date.\n<tool_call>\n{\"name\":\"calendar_search\",\"arguments\":{\"date\":\"2026-11-10\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calendar_search\",\"result\":\"No events found for 2026-11-10.\"}\n</tool_result>\nYou have no events scheduled for 2026-11-10."
    },
    # 2. Planning sequence of actions
    {
        "id": "agt_006",
        "user": "Find the weather in Paris and also tell me the local time there.",
        "assistant": "I will first get the weather for Paris and then the local time.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"Paris: 12°C, light rain, wind 15 km/h.\"}\n</tool_result>\n<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"Europe/Paris\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_time\",\"result\":\"Current time in Europe/Paris: 15:10.\"}\n</tool_result>\nIn Paris it is currently 12°C with light rain and a wind of 15 km/h. The local time is 15:10."
    },
    {
        "id": "agt_007",
        "user": "I need to remember to buy milk. Create a note about it.",
        "assistant": "I will create a note titled 'Shopping' with the content about buying milk.\n<tool_call>\n{\"name\":\"notes_create\",\"arguments\":{\"title\":\"Shopping\",\"content\":\"Buy milk\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_create\",\"result\":\"Note created successfully with id note_4821.\"}\n</tool_result>\nI have created a note titled 'Shopping' with the reminder to buy milk."
    },
    {
        "id": "agt_008",
        "user": "Search my notes for anything about the project deadline.",
        "assistant": "I will search your notes for the project deadline.\n<tool_call>\n{\"name\":\"notes_search\",\"arguments\":{\"query\":\"project deadline\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_search\",\"result\":\"Found 1 note: 'Project Alpha deadline is 2026-12-01'.\"}\n</tool_result>\nYour notes indicate that the Project Alpha deadline is 2026-12-01."
    },
    # 3. Structured tool calls + handling responses
    {
        "id": "agt_009",
        "user": "What is 87 divided by 3?",
        "assistant": "I will calculate 87 divided by 3.\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"87 / 3\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calculator\",\"result\":\"29\"}\n</tool_result>\n87 divided by 3 equals 29."
    },
    {
        "id": "agt_010",
        "user": "Look up information about the Great Barrier Reef.",
        "assistant": "I will search the web for information on the Great Barrier Reef.\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"Great Barrier Reef overview\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"web_search\",\"result\":\"The Great Barrier Reef is the world's largest coral reef system, located off the coast of Queensland, Australia. It is home to thousands of species and is a UNESCO World Heritage site.\"}\n</tool_result>\nThe Great Barrier Reef is the world's largest coral reef system off the coast of Queensland, Australia. It supports thousands of species and is a UNESCO World Heritage site."
    },
    {
        "id": "agt_011",
        "user": "Check the weather in Cape Town.",
        "assistant": "I will retrieve the weather for Cape Town.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Cape Town\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"Cape Town: 22°C, sunny, wind 10 km/h.\"}\n</tool_result>\nCape Town is currently 22°C and sunny with a wind of 10 km/h."
    },
    # 4. Detecting missing information
    {
        "id": "agt_012",
        "user": "What's the weather like?",
        "assistant": "I need a location to check the weather. Which city or place would you like the weather for?"
    },
    {
        "id": "agt_013",
        "user": "Set a reminder for me.",
        "assistant": "I can create a note for a reminder. What should the title and content of the reminder be?"
    },
    {
        "id": "agt_014",
        "user": "Search the calendar.",
        "assistant": "Please provide the date you would like me to search for in the calendar."
    },
    {
        "id": "agt_015",
        "user": "Calculate something for me.",
        "assistant": "What expression would you like me to calculate?"
    },
    # 5. Recovering from tool failures
    {
        "id": "agt_016",
        "user": "Get the weather for Atlantis.",
        "assistant": "I will try to get the weather for Atlantis.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Atlantis\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"Error: Location not found.\"}\n</tool_result>\nI could not find weather data for Atlantis. Please provide a real city or location name."
    },
    {
        "id": "agt_017",
        "user": "Search notes for 'secret password'.",
        "assistant": "I will search your notes for that phrase.\n<tool_call>\n{\"name\":\"notes_search\",\"arguments\":{\"query\":\"secret password\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_search\",\"result\":\"Error: No matching notes found.\"}\n</tool_result>\nNo notes matching 'secret password' were found."
    },
    {
        "id": "agt_018",
        "user": "What is the time in an invalid timezone called 'Foo/Bar'?",
        "assistant": "I will attempt to get the time for that timezone.\n<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"Foo/Bar\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_time\",\"result\":\"Error: Invalid timezone.\"}\n</tool_result>\nThe timezone 'Foo/Bar' is not recognised. Please provide a valid timezone such as 'America/New_York' or 'Asia/Tokyo'."
    },
    # 6. Multi-step task coordination
    {
        "id": "agt_019",
        "user": "Create a note titled 'Meeting Prep' with content 'Review slides', then search for any notes about meetings.",
        "assistant": "I will first create the note and then search for meeting-related notes.\n<tool_call>\n{\"name\":\"notes_create\",\"arguments\":{\"title\":\"Meeting Prep\",\"content\":\"Review slides\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_create\",\"result\":\"Note created successfully with id note_9102.\"}\n</tool_result>\n<tool_call>\n{\"name\":\"notes_search\",\"arguments\":{\"query\":\"meeting\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_search\",\"result\":\"Found 2 notes: 'Meeting Prep: Review slides'; 'Weekly meeting every Monday at 10 AM'.\"}\n</tool_result>\nI created the note 'Meeting Prep'. Searching for meeting notes returned two results: the new note about reviewing slides and an existing note about a weekly meeting on Mondays at 10 AM."
    },
    {
        "id": "agt_020",
        "user": "Calculate 45 + 17 and then search the web for the number of continents.",
        "assistant": "I will first perform the addition and then search for the number of continents.\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"45 + 17\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calculator\",\"result\":\"62\"}\n</tool_result>\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"number of continents on Earth\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"web_search\",\"result\":\"There are seven continents on Earth: Africa, Antarctica, Asia, Europe, North America, Oceania, and South America.\"}\n</tool_result>\n45 + 17 equals 62. There are seven continents on Earth."
    },
    # 7. Summarising completed actions
    {
        "id": "agt_021",
        "user": "Check the time in New York and the weather in New York, then summarise both.",
        "assistant": "I will retrieve the time and weather for New York.\n<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"America/New_York\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_time\",\"result\":\"Current time in America/New_York: 09:45.\"}\n</tool_result>\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"New York\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"New York: 8°C, overcast, humidity 70%.\"}\n</tool_result>\nSummary: It is currently 09:45 in New York. The weather is 8°C and overcast with 70% humidity."
    },
    {
        "id": "agt_022",
        "user": "Search for 'machine learning basics' and create a note with a short summary of the result.",
        "assistant": "I will search for machine learning basics and then create a note with a summary.\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"machine learning basics\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"web_search\",\"result\":\"Machine learning is a subset of AI where systems learn patterns from data to make predictions without being explicitly programmed for every case.\"}\n</tool_result>\n<tool_call>\n{\"name\":\"notes_create\",\"arguments\":{\"title\":\"ML Basics\",\"content\":\"Machine learning is a subset of AI where systems learn patterns from data to make predictions.\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_create\",\"result\":\"Note created successfully with id note_3340.\"}\n</tool_result>\nI searched for machine learning basics and created a note titled 'ML Basics' containing a short summary of the result."
    },
    # 8. Knowing when no tool is necessary
    {
        "id": "agt_023",
        "user": "What is 2 + 2?",
        "assistant": "2 + 2 equals 4."
    },
    {
        "id": "agt_024",
        "user": "Say hello in a friendly way.",
        "assistant": "Hello! How can I help you today?"
    },
    {
        "id": "agt_025",
        "user": "Is water wet?",
        "assistant": "Water is the substance that makes other things wet. By itself it is simply a liquid."
    },
    {
        "id": "agt_026",
        "user": "Thank you for your help.",
        "assistant": "You're welcome. Happy to help."
    },
    {
        "id": "agt_027",
        "user": "What is the opposite of 'start'?",
        "assistant": "The opposite of 'start' is 'end' or 'finish'."
    },
    # More agentic coverage
    {
        "id": "agt_028",
        "user": "Find out the weather in Sydney and save a note about it.",
        "assistant": "I will check the weather in Sydney and then create a note.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Sydney\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"Sydney: 25°C, clear skies, humidity 55%.\"}\n</tool_result>\n<tool_call>\n{\"name\":\"notes_create\",\"arguments\":{\"title\":\"Sydney Weather\",\"content\":\"25°C, clear skies, humidity 55%.\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_create\",\"result\":\"Note created successfully with id note_7715.\"}\n</tool_result>\nSydney is currently 25°C with clear skies. I have saved this information in a note titled 'Sydney Weather'."
    },
    {
        "id": "agt_029",
        "user": "What is the square root of 144?",
        "assistant": "I will calculate the square root of 144.\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"sqrt(144)\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calculator\",\"result\":\"12\"}\n</tool_result>\nThe square root of 144 is 12."
    },
    {
        "id": "agt_030",
        "user": "Search the web for the height of Mount Everest.",
        "assistant": "I will search for the height of Mount Everest.\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"height of Mount Everest\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"web_search\",\"result\":\"Mount Everest is 8,848.86 metres (29,031.7 feet) above sea level.\"}\n</tool_result>\nMount Everest is 8,848.86 metres (29,031.7 feet) above sea level."
    },
    {
        "id": "agt_031",
        "user": "Check my calendar for 2026-11-15.",
        "assistant": "I will search the calendar for 2026-11-15.\n<tool_call>\n{\"name\":\"calendar_search\",\"arguments\":{\"date\":\"2026-11-15\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calendar_search\",\"result\":\"Event found: Team sync at 11:00.\"}\n</tool_result>\nOn 2026-11-15 you have a team sync scheduled at 11:00."
    },
    {
        "id": "agt_032",
        "user": "Create a note called 'Ideas' with the text 'Explore new dataset formats'.",
        "assistant": "I will create that note now.\n<tool_call>\n{\"name\":\"notes_create\",\"arguments\":{\"title\":\"Ideas\",\"content\":\"Explore new dataset formats\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_create\",\"result\":\"Note created successfully with id note_2201.\"}\n</tool_result>\nThe note 'Ideas' has been created with the content you provided."
    },
    {
        "id": "agt_033",
        "user": "What time is it in Tokyo?",
        "assistant": "I will get the current time in Tokyo.\n<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"Asia/Tokyo\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_time\",\"result\":\"Current time in Asia/Tokyo: 22:15.\"}\n</tool_result>\nThe current time in Tokyo is 22:15."
    },
    {
        "id": "agt_034",
        "user": "I want the weather but I forgot the city name.",
        "assistant": "No problem. Please tell me the city or location for which you would like the weather."
    },
    {
        "id": "agt_035",
        "user": "Search notes for 'budget'.",
        "assistant": "I will search your notes for 'budget'.\n<tool_call>\n{\"name\":\"notes_search\",\"arguments\":{\"query\":\"budget\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_search\",\"result\":\"Found 1 note: 'Q4 budget review scheduled for next week'.\"}\n</tool_result>\nOne note mentions the Q4 budget review scheduled for next week."
    },
    {
        "id": "agt_036",
        "user": "Calculate 100 - 37.",
        "assistant": "I will subtract 37 from 100.\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"100 - 37\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calculator\",\"result\":\"63\"}\n</tool_result>\n100 minus 37 equals 63."
    },
    {
        "id": "agt_037",
        "user": "Get weather for a place called 'Nowhere City'.",
        "assistant": "I will attempt to retrieve weather for Nowhere City.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Nowhere City\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"Error: Location not found.\"}\n</tool_result>\nI could not find weather information for Nowhere City. Please try a different location name."
    },
    {
        "id": "agt_038",
        "user": "Tell me a fun fact about cats without using any tools.",
        "assistant": "Cats can make over 100 different vocal sounds, while dogs make only about 10."
    },
    {
        "id": "agt_039",
        "user": "Search the web for 'benefits of walking daily'.",
        "assistant": "I will search for the benefits of daily walking.\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"benefits of walking daily\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"web_search\",\"result\":\"Daily walking improves cardiovascular health, helps maintain a healthy weight, boosts mood, and strengthens bones and muscles.\"}\n</tool_result>\nDaily walking improves heart health, helps with weight control, lifts mood, and strengthens bones and muscles."
    },
    {
        "id": "agt_040",
        "user": "What is the time in an unknown zone 'XYZ/ABC'?",
        "assistant": "I will try to fetch the time for that timezone.\n<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"XYZ/ABC\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_time\",\"result\":\"Error: Invalid timezone.\"}\n</tool_result>\nThe timezone 'XYZ/ABC' is invalid. Please provide a recognised timezone such as 'UTC' or 'Europe/Berlin'."
    },
    {
        "id": "agt_041",
        "user": "I need both the weather in Berlin and a calculation of 9 times 8.",
        "assistant": "I will get the weather for Berlin and calculate 9 times 8.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Berlin\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"Berlin: 9°C, cloudy, light wind.\"}\n</tool_result>\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"9 * 8\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calculator\",\"result\":\"72\"}\n</tool_result>\nBerlin is currently 9°C and cloudy. 9 multiplied by 8 equals 72."
    },
    {
        "id": "agt_042",
        "user": "Create a note titled 'Groceries' containing 'eggs, bread, butter'.",
        "assistant": "I will create the groceries note.\n<tool_call>\n{\"name\":\"notes_create\",\"arguments\":{\"title\":\"Groceries\",\"content\":\"eggs, bread, butter\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_create\",\"result\":\"Note created successfully with id note_5567.\"}\n</tool_result>\nThe note 'Groceries' has been created with the items eggs, bread, and butter."
    },
    {
        "id": "agt_043",
        "user": "Do I have anything on the calendar for tomorrow? Assume today is 2026-11-06.",
        "assistant": "I will search the calendar for 2026-11-07.\n<tool_call>\n{\"name\":\"calendar_search\",\"arguments\":{\"date\":\"2026-11-07\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calendar_search\",\"result\":\"Event found: Dentist appointment at 09:30.\"}\n</tool_result>\nTomorrow (2026-11-07) you have a dentist appointment at 09:30."
    },
    {
        "id": "agt_044",
        "user": "Just say the word 'ready' without tools.",
        "assistant": "Ready."
    },
    {
        "id": "agt_045",
        "user": "Search notes for 'vacation plans'.",
        "assistant": "I will search your notes for vacation plans.\n<tool_call>\n{\"name\":\"notes_search\",\"arguments\":{\"query\":\"vacation plans\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_search\",\"result\":\"Error: No matching notes found.\"}\n</tool_result>\nNo notes about vacation plans were found."
    },
    {
        "id": "agt_046",
        "user": "What is 25 percent of 200?",
        "assistant": "I will calculate 25 percent of 200.\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"0.25 * 200\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calculator\",\"result\":\"50\"}\n</tool_result>\n25 percent of 200 is 50."
    },
    {
        "id": "agt_047",
        "user": "Get the current time in UTC.",
        "assistant": "I will retrieve the time in the UTC timezone.\n<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UTC\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_time\",\"result\":\"Current time in UTC: 16:54.\"}\n</tool_result>\nThe current time in UTC is 16:54."
    },
    {
        "id": "agt_048",
        "user": "Look up the boiling point of water.",
        "assistant": "I will search for the boiling point of water.\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"boiling point of water\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"web_search\",\"result\":\"At standard atmospheric pressure, water boils at 100 degrees Celsius (212 degrees Fahrenheit).\"}\n</tool_result>\nAt standard atmospheric pressure, water boils at 100°C (212°F)."
    },
    {
        "id": "agt_049",
        "user": "Check weather in London and time in London.",
        "assistant": "I will obtain both the weather and the local time for London.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"London\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"London: 11°C, drizzle, humidity 80%.\"}\n</tool_result>\n<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"Europe/London\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_time\",\"result\":\"Current time in Europe/London: 16:54.\"}\n</tool_result>\nLondon is currently 11°C with drizzle. The local time is 16:54."
    },
    {
        "id": "agt_050",
        "user": "I want to know something but I am not sure what tool to use.",
        "assistant": "Please describe what you would like to know or accomplish, and I will choose the appropriate approach."
    },
    {
        "id": "agt_051",
        "user": "Calculate 12 squared.",
        "assistant": "I will calculate 12 squared.\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"12 ** 2\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calculator\",\"result\":\"144\"}\n</tool_result>\n12 squared equals 144."
    },
    {
        "id": "agt_052",
        "user": "Create a note with title 'To Do' and content 'Finish dataset'.",
        "assistant": "I will create the requested note.\n<tool_call>\n{\"name\":\"notes_create\",\"arguments\":{\"title\":\"To Do\",\"content\":\"Finish dataset\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_create\",\"result\":\"Note created successfully with id note_8890.\"}\n</tool_result>\nNote 'To Do' has been created with the content 'Finish dataset'."
    },
    {
        "id": "agt_053",
        "user": "Search the web for 'largest desert in the world'.",
        "assistant": "I will search for the largest desert in the world.\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"largest desert in the world\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"web_search\",\"result\":\"The Antarctic Desert is the largest desert in the world by area, followed by the Arctic and the Sahara.\"}\n</tool_result>\nThe Antarctic Desert is the largest desert in the world by area."
    },
    {
        "id": "agt_054",
        "user": "What is the weather in a fictional place called 'Zorblax'?",
        "assistant": "I will try to get the weather for Zorblax.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Zorblax\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"Error: Location not found.\"}\n</tool_result>\nWeather data for Zorblax could not be found. Please provide a real location."
    },
    {
        "id": "agt_055",
        "user": "Tell me the capital of Japan without searching.",
        "assistant": "The capital of Japan is Tokyo."
    },
    {
        "id": "agt_056",
        "user": "Search calendar for 2026-12-25.",
        "assistant": "I will search the calendar for 2026-12-25.\n<tool_call>\n{\"name\":\"calendar_search\",\"arguments\":{\"date\":\"2026-12-25\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calendar_search\",\"result\":\"Event found: Holiday - Christmas Day.\"}\n</tool_result>\nOn 2026-12-25 the calendar shows a holiday for Christmas Day."
    },
    {
        "id": "agt_057",
        "user": "Find notes containing the word 'recipe'.",
        "assistant": "I will search your notes for 'recipe'.\n<tool_call>\n{\"name\":\"notes_search\",\"arguments\":{\"query\":\"recipe\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_search\",\"result\":\"Found 1 note: 'Pasta recipe: boil water, add pasta, cook 10 minutes'.\"}\n</tool_result>\nOne note contains a simple pasta recipe: boil water, add pasta, and cook for 10 minutes."
    },
    {
        "id": "agt_058",
        "user": "What is 7 times 6, and also what is the weather in Rome?",
        "assistant": "I will calculate 7 times 6 and retrieve the weather for Rome.\n<tool_call>\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"7 * 6\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"calculator\",\"result\":\"42\"}\n</tool_result>\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Rome\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_weather\",\"result\":\"Rome: 16°C, partly cloudy.\"}\n</tool_result>\n7 times 6 equals 42. The weather in Rome is 16°C and partly cloudy."
    },
    {
        "id": "agt_059",
        "user": "I need help but I have not specified the task.",
        "assistant": "Please tell me what you would like assistance with so I can help effectively."
    },
    {
        "id": "agt_060",
        "user": "Get the time in Sydney and create a note saying 'Checked Sydney time'.",
        "assistant": "I will first get the time in Sydney and then create the note.\n<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"Australia/Sydney\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"get_time\",\"result\":\"Current time in Australia/Sydney: 03:54.\"}\n</tool_result>\n<tool_call>\n{\"name\":\"notes_create\",\"arguments\":{\"title\":\"Time Check\",\"content\":\"Checked Sydney time\"}}\n</tool_call>\n<tool_result>\n{\"name\":\"notes_create\",\"result\":\"Note created successfully with id note_4412.\"}\n</tool_result>\nThe current time in Sydney is 03:54. I have also created a note titled 'Time Check' with the content you requested."
    },
]


def build_conversation(user: str, assistant: str) -> str:
    return (
        f"<system>\n{SYSTEM_PROMPT}\n</system>\n"
        f"<user>\n{user}\n</user>\n"
        f"<assistant>\n{assistant}\n</assistant>"
    )


def make_record(example: dict, category: str) -> dict:
    return {
        "id": example["id"],
        "category": category,
        "text": build_conversation(example["user"], example["assistant"]),
    }


def main():
    assert len(GENERAL_EXAMPLES) == 60, f"Expected 60 general, got {len(GENERAL_EXAMPLES)}"
    assert len(AGENTIC_EXAMPLES) == 60, f"Expected 60 agentic, got {len(AGENTIC_EXAMPLES)}"

    # Fixed deterministic splits (first 40 / next 10 / last 10 of each category)
    gen_train = GENERAL_EXAMPLES[:40]
    gen_val = GENERAL_EXAMPLES[40:50]
    gen_test = GENERAL_EXAMPLES[50:60]

    agt_train = AGENTIC_EXAMPLES[:40]
    agt_val = AGENTIC_EXAMPLES[40:50]
    agt_test = AGENTIC_EXAMPLES[50:60]

    train_records = [make_record(e, "general_language") for e in gen_train] + \
                    [make_record(e, "agentic") for e in agt_train]
    val_records = [make_record(e, "general_language") for e in gen_val] + \
                  [make_record(e, "agentic") for e in agt_val]
    test_records = [make_record(e, "general_language") for e in gen_test] + \
                   [make_record(e, "agentic") for e in agt_test]

    # Sort by id for reproducibility
    train_records.sort(key=lambda r: r["id"])
    val_records.sort(key=lambda r: r["id"])
    test_records.sort(key=lambda r: r["id"])

    def write_jsonl(path: Path, records: list):
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_jsonl(OUTPUT_DIR / "train.jsonl", train_records)
    write_jsonl(OUTPUT_DIR / "validation.jsonl", val_records)
    write_jsonl(OUTPUT_DIR / "test.jsonl", test_records)

    # Manifest
    all_ids = [r["id"] for r in train_records + val_records + test_records]
    manifest = {
        "dataset_name": "AYA Edge Mini Dataset",
        "dataset_version": "1.0.0",
        "total_examples": 120,
        "splits": {
            "train": 80,
            "validation": 20,
            "test": 20,
        },
        "categories": {
            "general_language": 60,
            "agentic": 60,
        },
        "category_distribution_per_split": {
            "train": {"general_language": 40, "agentic": 40},
            "validation": {"general_language": 10, "agentic": 10},
            "test": {"general_language": 10, "agentic": 10},
        },
        "generation_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "simulated_tools": [
            "web_search(query)",
            "calculator(expression)",
            "get_weather(location)",
            "get_time(timezone)",
            "calendar_search(date)",
            "notes_create(title, content)",
            "notes_search(query)",
        ],
        "unique_ids": len(set(all_ids)) == 120,
        "no_intentional_duplicates": True,
        "no_split_leakage": True,
        "description": (
            "Small synthetic dataset for overfitting experiments with AYA Edge 0.1 "
            "(85M parameter sparse MoE causal LM). Tests basic language pattern learning "
            "and simulated agentic tool-use behaviour. All tool results are simulated."
        ),
    }
    with (OUTPUT_DIR / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(train_records)} train, {len(val_records)} val, {len(test_records)} test.")
    print(f"Total unique IDs: {len(set(all_ids))}")
    print("Done.")


if __name__ == "__main__":
    main()
