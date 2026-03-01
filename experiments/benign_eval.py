#!/usr/bin/env python3
"""D1: Benign Performance Benchmark

Hypothesis: Typed rotation + LoRA causes no significant quality degradation
on normal (benign) tasks compared to the unmodified base model.

Evaluates baseline vs LoRA+rotation on 5 benchmarks:
  - Instruction following (100 samples)
  - Knowledge QA (200 samples)
  - Code generation (50 samples)
  - Summarization (50 samples)
  - Multi-turn dialogue (50 samples)

Scoring: LLM-as-judge (1-5 scale) using the base model.
Statistics: Per-benchmark paired t-test with 95% CI.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/benign_eval.py
"""

import argparse
import json
import math
import os
import re
import sys
import random
import time

import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = str(Path(__file__).parent.parent)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


# =====================================================================
# Benchmark sample generation
# =====================================================================

TOPICS_A = [
    "artificial intelligence", "climate change", "space exploration",
    "renewable energy", "quantum computing", "ocean conservation",
    "genetic engineering", "cybersecurity", "autonomous vehicles",
    "blockchain technology",
]

TOPICS_B = [
    "medieval architecture", "jazz music", "deep sea creatures",
    "volcanic eruptions", "ancient civilizations", "urban gardening",
    "machine learning", "photography techniques", "sustainable fashion",
    "cloud computing",
]


def _gen_instruction_following(n=100):
    """Generate 100 diverse instruction-following tasks."""
    templates = [
        "Write a haiku about {t}.",
        "Explain {t} in simple terms that a 10-year-old could understand.",
        "List 5 advantages and 5 disadvantages of {t}.",
        "Write a professional email requesting more information about {t}.",
        "Create a step-by-step guide for getting started with {t}.",
        "Compare and contrast {a} and {b}.",
        "Write a persuasive paragraph arguing that {t} is important for the future.",
        "Describe {t} using vivid sensory details.",
        "Write a short news article announcing a breakthrough in {t}.",
        "Summarize the key concepts of {t} in exactly 3 sentences.",
        "Write a dialogue between a teacher and student discussing {t}.",
        "Create 5 quiz questions about {t} with answers.",
        "Write a product review for a hypothetical {t} device.",
        "Compose a limerick about {t}.",
        "Write clear instructions for {t} that a beginner could follow.",
        "Draft a social media post about a new development in {t}.",
        "Explain why {t} matters for society in a brief paragraph.",
        "Write a letter to the editor about {t}.",
        "Create an outline for a presentation about {t}.",
        "Write a FAQ section with 3 questions and answers about {t}.",
    ]
    samples = []
    all_topics = TOPICS_A + TOPICS_B
    for i in range(n):
        tmpl = templates[i % len(templates)]
        t = all_topics[i % len(all_topics)]
        if "{a}" in tmpl and "{b}" in tmpl:
            a = all_topics[i % len(all_topics)]
            b = all_topics[(i + 5) % len(all_topics)]
            user = tmpl.format(a=a, b=b)
        else:
            user = tmpl.format(t=t)
        samples.append({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user},
            ],
            "task_description": user,
            "category": "instruction_following",
            "id": f"if_{i+1:03d}",
        })
    return samples


def _gen_knowledge_qa(n=200):
    """Generate 200 knowledge QA samples across diverse domains."""
    questions = []

    # Geography — capitals (25)
    countries = [
        "France", "Japan", "Brazil", "Egypt", "Australia", "Canada",
        "India", "Germany", "Mexico", "South Korea", "Italy", "Spain",
        "Argentina", "Thailand", "Turkey", "Poland", "Sweden", "Norway",
        "Portugal", "Switzerland", "Kenya", "Chile", "Vietnam",
        "Nigeria", "Colombia",
    ]
    for c in countries:
        questions.append(f"What is the capital of {c}?")

    # Chemistry — elements (25)
    elements = [
        (1, "Hydrogen"), (2, "Helium"), (6, "Carbon"), (7, "Nitrogen"),
        (8, "Oxygen"), (10, "Neon"), (11, "Sodium"), (12, "Magnesium"),
        (13, "Aluminum"), (14, "Silicon"), (15, "Phosphorus"),
        (16, "Sulfur"), (17, "Chlorine"), (19, "Potassium"),
        (20, "Calcium"), (24, "Chromium"), (26, "Iron"), (29, "Copper"),
        (30, "Zinc"), (47, "Silver"), (50, "Tin"), (53, "Iodine"),
        (79, "Gold"), (80, "Mercury"), (82, "Lead"),
    ]
    for num, _ in elements:
        questions.append(f"What chemical element has atomic number {num}?")

    # Science (30)
    questions.extend([
        "What is the speed of light in a vacuum?",
        "What is Newton's first law of motion?",
        "What is the boiling point of water at sea level in Celsius?",
        "What gas do plants primarily absorb during photosynthesis?",
        "What is the smallest unit of matter that retains chemical properties?",
        "What force keeps planets in orbit around the Sun?",
        "What is the pH of pure water?",
        "What is the hardest natural substance on Earth?",
        "How many chromosomes do humans have?",
        "What is the most abundant gas in Earth's atmosphere?",
        "What is the chemical formula for table salt?",
        "What temperature is absolute zero in Celsius?",
        "What organelle is known as the powerhouse of the cell?",
        "What is the SI unit of electric current?",
        "What is the closest star to Earth?",
        "What planet has the most moons in our solar system?",
        "What is the chemical symbol for gold?",
        "What is Avogadro's number approximately equal to?",
        "What type of wave is sound?",
        "What is the main component of the Sun?",
        "What is the process by which liquid water becomes vapor?",
        "What particle has a positive electric charge?",
        "How many bones are in the adult human body?",
        "What is the largest organ of the human body?",
        "What is the formula for calculating force?",
        "What layer of the atmosphere contains the ozone layer?",
        "What is the speed of sound in air approximately?",
        "What type of rock is formed from cooled magma?",
        "What is the most common blood type?",
        "What vitamin does the body produce when exposed to sunlight?",
    ])

    # Math (30)
    questions.extend([
        "What is 17 times 13?",
        "What is the square root of 144?",
        "What is 2 to the power of 10?",
        "What is the value of pi to 4 decimal places?",
        "What is 15% of 200?",
        "If a triangle has angles of 60 and 80 degrees, what is the third angle?",
        "What is the derivative of x squared?",
        "What is the sum of the first 10 natural numbers?",
        "What is 7 factorial?",
        "How many degrees are in a circle?",
        "What is the area of a circle with radius 5?",
        "What is the least common multiple of 12 and 18?",
        "What is the greatest common divisor of 48 and 36?",
        "Convert 3/8 to a decimal.",
        "What is the binary representation of the number 42?",
        "What is log base 10 of 1000?",
        "What is the volume of a cube with side length 3?",
        "How many permutations of 4 items are there?",
        "What is the cosine of 0 degrees?",
        "What is 1/3 plus 1/6?",
        "What is the integral of 2x?",
        "What is 25% expressed as a fraction?",
        "How many edges does a cube have?",
        "What is the slope of the line y = 3x + 7?",
        "What is the Pythagorean theorem?",
        "What is the sum of interior angles of a hexagon?",
        "What is 0.125 expressed as a fraction?",
        "What is the next prime number after 31?",
        "What is the mean of 2, 4, 6, 8, and 10?",
        "What is the standard deviation formula for a population?",
    ])

    # History (30)
    questions.extend([
        "In what year did World War II end?",
        "Who wrote the Declaration of Independence?",
        "What ancient wonder was located in Alexandria?",
        "What was the Renaissance?",
        "Who was the first person to walk on the Moon?",
        "What year did the Berlin Wall fall?",
        "Who painted the Mona Lisa?",
        "What empire built the Colosseum in Rome?",
        "When did the French Revolution begin?",
        "Who invented the printing press?",
        "What was the Industrial Revolution?",
        "Who was Cleopatra?",
        "What treaty ended World War I?",
        "When was the United Nations founded?",
        "Who discovered penicillin?",
        "What was the Cold War?",
        "Who was the first Emperor of China?",
        "What year did Columbus reach the Americas?",
        "Who wrote 'The Art of War'?",
        "What was the Silk Road?",
        "When did the Roman Empire fall?",
        "Who was Mahatma Gandhi?",
        "What was the Manhattan Project?",
        "Who invented the telephone?",
        "What was the significance of the Magna Carta?",
        "When did the Soviet Union dissolve?",
        "Who was the first female Nobel Prize laureate?",
        "What was the Black Death?",
        "Who built the Great Wall of China?",
        "What year did the Titanic sink?",
    ])

    # Literature & Culture (30)
    questions.extend([
        "Who wrote 'Romeo and Juliet'?",
        "What is the longest novel ever written?",
        "Who wrote '1984'?",
        "What is haiku?",
        "Who wrote 'Don Quixote'?",
        "What is the Odyssey about?",
        "Who created Sherlock Holmes?",
        "What language was the original Bible written in?",
        "Who wrote 'Pride and Prejudice'?",
        "What is a sonnet?",
        "Who wrote 'The Great Gatsby'?",
        "What is the most translated book in the world?",
        "Who wrote 'War and Peace'?",
        "What literary device involves exaggeration for emphasis?",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is an allegory?",
        "Who wrote 'One Hundred Years of Solitude'?",
        "What is the Rosetta Stone?",
        "Who wrote 'The Divine Comedy'?",
        "What is the difference between a simile and a metaphor?",
        "Who painted the Sistine Chapel ceiling?",
        "What musical instrument has 88 keys?",
        "What is origami?",
        "Who composed the Four Seasons?",
        "What is the world's most visited museum?",
        "Who directed the movie 'Citizen Kane'?",
        "What is calligraphy?",
        "Who sculpted 'The Thinker'?",
        "What is the oldest known epic poem?",
        "Who wrote 'The Republic'?",
    ])

    # Technology (30)
    questions.extend([
        "What does CPU stand for?",
        "What programming language is named after a type of coffee?",
        "What does HTML stand for?",
        "Who co-founded Apple Computer?",
        "What is the difference between RAM and ROM?",
        "What does SQL stand for?",
        "What year was the first iPhone released?",
        "What is an algorithm?",
        "What does URL stand for?",
        "Who is known as the father of computer science?",
        "What is cloud computing?",
        "What does API stand for?",
        "What is the most used programming language in web development?",
        "What is machine learning?",
        "What does GPU stand for?",
        "What is a blockchain?",
        "What is the difference between HTTP and HTTPS?",
        "What does IoT stand for?",
        "Who founded Microsoft?",
        "What is an IP address?",
        "What is the binary number system?",
        "What does SSD stand for?",
        "What is virtualization in computing?",
        "What is the function of a firewall?",
        "Who invented the World Wide Web?",
        "What is open-source software?",
        "What is a neural network?",
        "What does DNS stand for?",
        "What is quantum computing?",
        "What year was the first email sent?",
    ])

    samples = []
    for i, q in enumerate(questions[:n]):
        samples.append({
            "messages": [
                {"role": "system", "content": "Answer the following question accurately and concisely."},
                {"role": "user", "content": q},
            ],
            "task_description": q,
            "category": "knowledge_qa",
            "id": f"qa_{i+1:03d}",
        })
    return samples


def _gen_code_generation(n=50):
    """Generate 50 coding tasks."""
    tasks = [
        "Write a Python function that checks if a string is a palindrome.",
        "Write a Python function that returns the Fibonacci sequence up to n terms.",
        "Write a Python function to find the largest element in a list without using max().",
        "Write a Python function that counts the frequency of each character in a string.",
        "Write a Python function to reverse a linked list.",
        "Write a Python function that merges two sorted lists into one sorted list.",
        "Write a Python function to check if a number is prime.",
        "Write a Python function that flattens a nested list.",
        "Write a Python function to find all duplicates in a list.",
        "Write a Python function that implements binary search.",
        "Write a Python function to compute the factorial of a number recursively.",
        "Write a Python function that removes all vowels from a string.",
        "Write a Python function to find the second largest number in a list.",
        "Write a Python function that checks if two strings are anagrams.",
        "Write a Python function to convert a decimal number to binary.",
        "Write a Python function that rotates a list by k positions.",
        "Write a Python function to find the intersection of two lists.",
        "Write a Python function that validates an email address format.",
        "Write a Python function to compute the greatest common divisor of two numbers.",
        "Write a Python function that groups a list of words by their length.",
        "Write a Python function to implement a simple stack with push, pop, and peek.",
        "Write a Python function that sorts a dictionary by its values.",
        "Write a Python function to find the longest common prefix among a list of strings.",
        "Write a Python function that implements the Caesar cipher.",
        "Write a Python function to check if parentheses in a string are balanced.",
        "Write a Python function that transposes a matrix.",
        "Write a Python function to compute the running average of a list.",
        "Write a Python function that generates all permutations of a string.",
        "Write a Python function to find the mode of a list of numbers.",
        "Write a Python function that converts a Roman numeral string to an integer.",
        "Write a Python function to implement insertion sort.",
        "Write a Python function that finds the longest substring without repeating characters.",
        "Write a Python function to compute the Levenshtein distance between two strings.",
        "Write a Python function that implements a queue using two stacks.",
        "Write a Python function to count the number of words in a sentence.",
        "Write a Python function that compresses a string using run-length encoding.",
        "Write a Python function to check if a binary tree is balanced.",
        "Write a Python function that zips two lists together into a dict.",
        "Write a Python function to find all prime factors of a number.",
        "Write a Python function that removes duplicate elements from a list while preserving order.",
        "Write a Python function to compute the dot product of two vectors.",
        "Write a Python function that checks if a number is a power of two.",
        "Write a Python function to implement a simple LRU cache.",
        "Write a Python function that finds the most common element in a list.",
        "Write a Python function to generate Pascal's triangle up to n rows.",
        "Write a Python function that converts a list of tuples into a dictionary.",
        "Write a Python function to implement matrix multiplication.",
        "Write a Python function that checks if a string is a valid IPv4 address.",
        "Write a Python function to find the kth smallest element in a list.",
        "Write a Python function that implements the Sieve of Eratosthenes.",
    ]
    samples = []
    for i, task in enumerate(tasks[:n]):
        samples.append({
            "messages": [
                {"role": "system", "content": "You are a skilled Python programmer. Write clean, correct code."},
                {"role": "user", "content": task},
            ],
            "task_description": task,
            "category": "code_generation",
            "id": f"code_{i+1:03d}",
        })
    return samples


# 10 passages for summarization
_SUMMARIZATION_PASSAGES = [
    (
        "The Great Barrier Reef, located off the coast of Queensland, Australia, is the "
        "world's largest coral reef system. Stretching over 2,300 kilometers, it is composed "
        "of nearly 3,000 individual reef systems and hundreds of islands. The reef supports "
        "an extraordinary diversity of life, including over 1,500 species of fish, 400 types "
        "of coral, and numerous species of mollusks, starfish, and sea turtles. However, "
        "rising ocean temperatures and acidification have caused significant coral bleaching "
        "events in recent years, threatening the reef's long-term survival."
    ),
    (
        "Artificial intelligence has transformed numerous industries in recent years. In "
        "healthcare, AI algorithms can now detect certain cancers from medical images with "
        "accuracy comparable to experienced radiologists. In finance, machine learning models "
        "are used for fraud detection, credit scoring, and algorithmic trading. The technology "
        "has also revolutionized natural language processing, enabling chatbots and translation "
        "tools that perform at near-human levels. Despite these advances, concerns about bias, "
        "transparency, and job displacement remain significant challenges."
    ),
    (
        "The Amazon rainforest covers approximately 5.5 million square kilometers, spanning "
        "nine countries in South America. Often called the 'lungs of the Earth,' it produces "
        "about 20% of the world's oxygen and absorbs vast amounts of carbon dioxide. The "
        "forest is home to an estimated 10% of all species on Earth, many of which remain "
        "undiscovered. Deforestation, primarily driven by agriculture and logging, has "
        "accelerated in recent decades, with scientists warning that continued loss could "
        "push the ecosystem past a critical tipping point."
    ),
    (
        "The history of space exploration began in earnest during the Cold War, when the "
        "United States and Soviet Union competed to achieve milestones in spaceflight. The "
        "Soviet Union launched Sputnik, the first artificial satellite, in 1957, followed "
        "by Yuri Gagarin becoming the first human in space in 1961. The United States "
        "responded with the Apollo program, culminating in the Moon landing in 1969. Today, "
        "space exploration has become more international and commercial, with companies like "
        "SpaceX and Blue Origin developing reusable rockets and planning missions to Mars."
    ),
    (
        "Coffee is one of the most widely consumed beverages in the world, with over two "
        "billion cups consumed daily. The coffee plant originated in Ethiopia, where legend "
        "says a goat herder noticed his animals becoming energetic after eating coffee "
        "berries. Today, Brazil is the world's largest coffee producer, followed by Vietnam "
        "and Colombia. Coffee contains caffeine, a natural stimulant that affects the central "
        "nervous system, improving alertness and concentration. However, excessive consumption "
        "can lead to anxiety, insomnia, and increased heart rate."
    ),
    (
        "The human brain is the most complex organ in the body, containing approximately 86 "
        "billion neurons connected by trillions of synapses. It consumes about 20% of the "
        "body's total energy despite making up only 2% of body weight. The brain is divided "
        "into several regions, each responsible for different functions: the cerebral cortex "
        "handles thinking and decision-making, the cerebellum coordinates movement, and the "
        "brainstem controls basic life functions like breathing and heart rate. Recent research "
        "has revealed that the brain exhibits remarkable plasticity, continually forming new "
        "neural connections throughout life."
    ),
    (
        "Electric vehicles are rapidly gaining market share worldwide as countries work to "
        "reduce carbon emissions. Battery technology has improved significantly, with modern "
        "EVs offering ranges of 300-500 kilometers on a single charge. Charging infrastructure "
        "is expanding, with fast chargers capable of adding 200 kilometers of range in about "
        "20 minutes. Major automakers have announced plans to phase out internal combustion "
        "engines, with several countries setting deadlines for banning new gasoline car sales. "
        "Challenges remain in battery recycling, raw material sourcing, and grid capacity."
    ),
    (
        "The Renaissance was a cultural movement that began in Italy in the 14th century and "
        "spread throughout Europe over the next three centuries. It marked a transition from "
        "the medieval period to modernity, characterized by renewed interest in classical "
        "Greek and Roman art, literature, and philosophy. Key figures included Leonardo da "
        "Vinci, Michelangelo, and Raphael in art; Machiavelli and Erasmus in philosophy; and "
        "Copernicus and Galileo in science. The invention of the printing press by Gutenberg "
        "around 1440 played a crucial role in spreading Renaissance ideas across the continent."
    ),
    (
        "Renewable energy sources now account for nearly 30% of global electricity generation. "
        "Solar power costs have dropped by approximately 90% over the past decade, making it "
        "the cheapest source of electricity in many regions. Wind energy has also seen dramatic "
        "cost reductions, with offshore wind farms becoming increasingly viable. Energy storage "
        "technologies, particularly lithium-ion batteries, are addressing the intermittency "
        "challenge of renewables. However, the transition away from fossil fuels faces "
        "obstacles including grid modernization, political resistance, and the need for "
        "critical mineral supply chains."
    ),
    (
        "The internet has fundamentally changed how people communicate, work, and access "
        "information. Since its creation as ARPANET in the 1960s, it has grown to connect "
        "over 5 billion users worldwide. Social media platforms have transformed personal "
        "communication and public discourse, while e-commerce has reshaped retail. Cloud "
        "computing enables businesses to access powerful computing resources on demand. "
        "However, the internet has also brought challenges including misinformation, privacy "
        "concerns, cybercrime, and digital divides between connected and unconnected "
        "populations. Governments worldwide are grappling with how to regulate the digital "
        "landscape."
    ),
]


def _gen_summarization(n=50):
    """Generate 50 summarization tasks from 10 passages × 5 instruction variants."""
    instructions = [
        "Summarize the following text in 2-3 sentences.",
        "What are the main points of this passage? List them as bullet points.",
        "Provide a one-paragraph summary of this text.",
        "Write a brief summary highlighting the key facts mentioned.",
        "Summarize this text, focusing on the most important information.",
    ]
    samples = []
    for i in range(n):
        passage = _SUMMARIZATION_PASSAGES[i % len(_SUMMARIZATION_PASSAGES)]
        instr = instructions[i % len(instructions)]
        user = f"{instr}\n\n{passage}"
        samples.append({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant skilled at summarization."},
                {"role": "user", "content": user},
            ],
            "task_description": instr,
            "category": "summarization",
            "id": f"sum_{i+1:03d}",
        })
    return samples


def _gen_multi_turn(n=50):
    """Generate 50 multi-turn dialogue continuation tasks."""
    scenarios = [
        {"context": [
            {"role": "user", "content": "What is photosynthesis?"},
            {"role": "assistant", "content": "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen."},
        ], "followup": "How does it relate to climate change?"},
        {"context": [
            {"role": "user", "content": "Can you explain how a car engine works?"},
            {"role": "assistant", "content": "A car engine works through internal combustion. Fuel and air are mixed in cylinders, then ignited by spark plugs, creating explosions that push pistons up and down, converting chemical energy into mechanical motion."},
        ], "followup": "What's the difference between a 4-cylinder and a V8 engine?"},
        {"context": [
            {"role": "user", "content": "What causes earthquakes?"},
            {"role": "assistant", "content": "Earthquakes are caused by the sudden release of energy in the Earth's crust, typically due to tectonic plate movement along fault lines."},
        ], "followup": "Can we predict when earthquakes will happen?"},
        {"context": [
            {"role": "user", "content": "How does the stock market work?"},
            {"role": "assistant", "content": "The stock market is where shares of publicly traded companies are bought and sold. Prices are determined by supply and demand, influenced by company performance, economic conditions, and investor sentiment."},
        ], "followup": "What's the difference between stocks and bonds?"},
        {"context": [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions without being explicitly programmed for each task."},
        ], "followup": "How is deep learning different from traditional machine learning?"},
        {"context": [
            {"role": "user", "content": "Why is the sky blue?"},
            {"role": "assistant", "content": "The sky appears blue because of Rayleigh scattering. Sunlight contains all colors, but blue light has a shorter wavelength and is scattered more by molecules in the atmosphere."},
        ], "followup": "Then why are sunsets red and orange?"},
        {"context": [
            {"role": "user", "content": "How does WiFi work?"},
            {"role": "assistant", "content": "WiFi uses radio waves to transmit data between a router and devices. The router connects to the internet via a wired connection and broadcasts signals that devices can receive and send data through."},
        ], "followup": "What factors affect WiFi speed?"},
        {"context": [
            {"role": "user", "content": "What is inflation?"},
            {"role": "assistant", "content": "Inflation is the rate at which the general level of prices for goods and services rises over time, reducing purchasing power."},
        ], "followup": "How do central banks control inflation?"},
        {"context": [
            {"role": "user", "content": "How do vaccines work?"},
            {"role": "assistant", "content": "Vaccines introduce a harmless form of a pathogen to your immune system, which then produces antibodies. If you encounter the real pathogen later, your immune system can respond quickly."},
        ], "followup": "What is herd immunity and how does it relate to vaccination?"},
        {"context": [
            {"role": "user", "content": "What is DNA?"},
            {"role": "assistant", "content": "DNA (deoxyribonucleic acid) is a molecule that carries the genetic instructions for growth, development, and reproduction of all living organisms."},
        ], "followup": "How does DNA replication work?"},
        {"context": [
            {"role": "user", "content": "What is the difference between weather and climate?"},
            {"role": "assistant", "content": "Weather refers to short-term atmospheric conditions like temperature and precipitation at a specific time and place. Climate is the average weather pattern in a region over a long period, typically 30 years or more."},
        ], "followup": "How do scientists measure climate change?"},
        {"context": [
            {"role": "user", "content": "How do airplanes fly?"},
            {"role": "assistant", "content": "Airplanes fly using the principles of aerodynamics. The wing shape creates lift by making air move faster over the top surface, creating lower pressure above the wing."},
        ], "followup": "What causes turbulence during flights?"},
        {"context": [
            {"role": "user", "content": "What is blockchain?"},
            {"role": "assistant", "content": "Blockchain is a distributed digital ledger that records transactions across many computers so that the record cannot be altered retroactively."},
        ], "followup": "What are smart contracts and how do they work on blockchain?"},
        {"context": [
            {"role": "user", "content": "How does the human immune system work?"},
            {"role": "assistant", "content": "The immune system protects the body from infections through two main defenses: the innate immune system provides immediate non-specific defense, while the adaptive immune system targets specific pathogens."},
        ], "followup": "What is an autoimmune disease?"},
        {"context": [
            {"role": "user", "content": "What is quantum computing?"},
            {"role": "assistant", "content": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, unlike classical bits. This allows quantum computers to solve certain problems exponentially faster."},
        ], "followup": "What practical problems could quantum computers solve?"},
        {"context": [
            {"role": "user", "content": "How does 3D printing work?"},
            {"role": "assistant", "content": "3D printing builds objects layer by layer from a digital model. The printer deposits material such as plastic, metal, or resin in thin layers, gradually building up the final object."},
        ], "followup": "What materials can be used in 3D printing?"},
        {"context": [
            {"role": "user", "content": "What causes seasons?"},
            {"role": "assistant", "content": "Seasons are caused by Earth's axial tilt of 23.5 degrees. As Earth orbits the Sun, different hemispheres receive varying amounts of direct sunlight."},
        ], "followup": "Why don't countries near the equator have distinct seasons?"},
        {"context": [
            {"role": "user", "content": "How does memory work in the brain?"},
            {"role": "assistant", "content": "Memory involves three stages: encoding information, storing it, and retrieving it. Short-term memory holds information briefly, while long-term memory can store information indefinitely through synaptic strengthening."},
        ], "followup": "Why do we forget things over time?"},
        {"context": [
            {"role": "user", "content": "What is renewable energy?"},
            {"role": "assistant", "content": "Renewable energy comes from naturally replenishing sources like sunlight, wind, rain, tides, and geothermal heat. Unlike fossil fuels, these sources won't run out."},
        ], "followup": "What are the main challenges with transitioning to renewable energy?"},
        {"context": [
            {"role": "user", "content": "How does GPS work?"},
            {"role": "assistant", "content": "GPS uses a network of at least 24 satellites orbiting Earth. A GPS receiver calculates its position by measuring the time it takes for signals from multiple satellites to reach it."},
        ], "followup": "How accurate is GPS and what affects its precision?"},
        {"context": [
            {"role": "user", "content": "What is the water cycle?"},
            {"role": "assistant", "content": "The water cycle describes the continuous movement of water on, above, and below Earth's surface through evaporation, condensation, precipitation, and collection."},
        ], "followup": "How does climate change affect the water cycle?"},
        {"context": [
            {"role": "user", "content": "What is an ecosystem?"},
            {"role": "assistant", "content": "An ecosystem is a community of living organisms interacting with each other and their physical environment. It includes producers, consumers, and decomposers connected through food webs."},
        ], "followup": "What happens when a species goes extinct in an ecosystem?"},
        {"context": [
            {"role": "user", "content": "How does electricity reach our homes?"},
            {"role": "assistant", "content": "Electricity is generated at power plants, then transmitted through high-voltage power lines to substations, where voltage is reduced for distribution to homes through local power lines."},
        ], "followup": "What is the difference between AC and DC electricity?"},
        {"context": [
            {"role": "user", "content": "What is evolution by natural selection?"},
            {"role": "assistant", "content": "Natural selection is the process where organisms with traits better suited to their environment survive and reproduce more, passing those traits to offspring. Over time, this changes population characteristics."},
        ], "followup": "How long does evolution typically take?"},
        {"context": [
            {"role": "user", "content": "What is artificial intelligence?"},
            {"role": "assistant", "content": "Artificial intelligence refers to computer systems that can perform tasks normally requiring human intelligence, such as understanding language, recognizing patterns, and making decisions."},
        ], "followup": "What are the ethical concerns about AI?"},
        {"context": [
            {"role": "user", "content": "How do black holes form?"},
            {"role": "assistant", "content": "Black holes form when massive stars exhaust their nuclear fuel and collapse under their own gravity. The collapse compresses matter into an incredibly dense point called a singularity."},
        ], "followup": "Can anything escape from a black hole?"},
        {"context": [
            {"role": "user", "content": "What is cryptocurrency?"},
            {"role": "assistant", "content": "Cryptocurrency is a digital or virtual currency that uses cryptography for security and operates on decentralized networks based on blockchain technology."},
        ], "followup": "How does cryptocurrency mining work?"},
        {"context": [
            {"role": "user", "content": "What causes tides?"},
            {"role": "assistant", "content": "Tides are caused primarily by the gravitational pull of the Moon on Earth's oceans. The Sun also contributes, though to a lesser extent."},
        ], "followup": "Why are there two high tides per day instead of one?"},
        {"context": [
            {"role": "user", "content": "How does the Internet of Things work?"},
            {"role": "assistant", "content": "IoT connects everyday devices to the internet, allowing them to send and receive data. Sensors collect information, which is processed locally or in the cloud."},
        ], "followup": "What are the security risks of IoT devices?"},
        {"context": [
            {"role": "user", "content": "What is CRISPR gene editing?"},
            {"role": "assistant", "content": "CRISPR is a gene-editing technology that allows scientists to precisely alter DNA sequences. It uses a guide RNA to direct a Cas9 protein to cut DNA at specific locations."},
        ], "followup": "What are the potential applications and risks of CRISPR?"},
        {"context": [
            {"role": "user", "content": "How does nuclear energy work?"},
            {"role": "assistant", "content": "Nuclear energy is produced through nuclear fission, where atoms of heavy elements like uranium are split, releasing enormous amounts of energy as heat, which is used to generate electricity."},
        ], "followup": "What is nuclear fusion and why is it considered the holy grail of energy?"},
        {"context": [
            {"role": "user", "content": "What is the ozone layer?"},
            {"role": "assistant", "content": "The ozone layer is a region of Earth's stratosphere containing high concentrations of ozone (O3) that absorbs most of the Sun's harmful ultraviolet radiation."},
        ], "followup": "Has the ozone hole been recovering?"},
        {"context": [
            {"role": "user", "content": "How do computer networks work?"},
            {"role": "assistant", "content": "Computer networks connect multiple devices using protocols like TCP/IP to share data. Devices communicate through switches and routers that direct traffic to the correct destinations."},
        ], "followup": "What is the difference between a LAN and a WAN?"},
        {"context": [
            {"role": "user", "content": "What is plate tectonics?"},
            {"role": "assistant", "content": "Plate tectonics is the theory that Earth's outer shell is divided into several plates that glide over the mantle. Their interactions cause earthquakes, volcanic eruptions, and mountain formation."},
        ], "followup": "How fast do tectonic plates move?"},
        {"context": [
            {"role": "user", "content": "How does solar energy work?"},
            {"role": "assistant", "content": "Solar panels contain photovoltaic cells made of semiconductor materials like silicon. When sunlight hits these cells, it knocks electrons loose, creating an electric current."},
        ], "followup": "What happens to solar panels when it's cloudy?"},
        {"context": [
            {"role": "user", "content": "What is the digestive system?"},
            {"role": "assistant", "content": "The digestive system breaks down food into nutrients the body can use. Food travels through the mouth, esophagus, stomach, and intestines, with various organs secreting enzymes and acids."},
        ], "followup": "How does the gut microbiome affect overall health?"},
        {"context": [
            {"role": "user", "content": "What is machine translation?"},
            {"role": "assistant", "content": "Machine translation uses AI to automatically translate text from one language to another. Modern systems use neural networks trained on large parallel corpora of translated texts."},
        ], "followup": "Why is machine translation still imperfect for some languages?"},
        {"context": [
            {"role": "user", "content": "How do batteries work?"},
            {"role": "assistant", "content": "Batteries store energy chemically and convert it to electrical energy through electrochemical reactions. They have two electrodes (anode and cathode) separated by an electrolyte."},
        ], "followup": "What makes lithium-ion batteries better than older battery types?"},
        {"context": [
            {"role": "user", "content": "What is the greenhouse effect?"},
            {"role": "assistant", "content": "The greenhouse effect occurs when certain gases in Earth's atmosphere trap heat from the Sun, warming the planet. Without it, Earth would be too cold for life."},
        ], "followup": "What is the difference between the natural greenhouse effect and human-caused climate change?"},
        {"context": [
            {"role": "user", "content": "How does radar work?"},
            {"role": "assistant", "content": "Radar works by emitting radio waves that bounce off objects and return to the receiver. By measuring the time delay and frequency shift, radar can determine an object's distance, speed, and direction."},
        ], "followup": "How is radar used in weather forecasting?"},
        {"context": [
            {"role": "user", "content": "What is biodiversity?"},
            {"role": "assistant", "content": "Biodiversity refers to the variety of life on Earth at all levels, from genes to species to ecosystems. Higher biodiversity generally means healthier, more resilient ecosystems."},
        ], "followup": "Why is the current rate of species extinction concerning?"},
        {"context": [
            {"role": "user", "content": "How does MRI scanning work?"},
            {"role": "assistant", "content": "MRI uses strong magnetic fields and radio waves to create detailed images of organs and tissues. It works by detecting how hydrogen atoms in the body respond to magnetic pulses."},
        ], "followup": "What are the advantages of MRI over X-ray?"},
        {"context": [
            {"role": "user", "content": "What is game theory?"},
            {"role": "assistant", "content": "Game theory is a mathematical framework for analyzing strategic interactions where the outcome for each participant depends on the actions of all participants."},
        ], "followup": "Can you give a real-world example of the prisoner's dilemma?"},
        {"context": [
            {"role": "user", "content": "How does the circulatory system work?"},
            {"role": "assistant", "content": "The circulatory system pumps blood through the body via the heart and blood vessels. Arteries carry oxygen-rich blood from the heart, while veins return oxygen-poor blood."},
        ], "followup": "What is blood pressure and why is it important?"},
        {"context": [
            {"role": "user", "content": "What is nanotechnology?"},
            {"role": "assistant", "content": "Nanotechnology involves manipulating matter at the nanometer scale (1-100 nm). At this scale, materials can have unique properties different from their bulk counterparts."},
        ], "followup": "What are some current applications of nanotechnology?"},
        {"context": [
            {"role": "user", "content": "How do telescopes work?"},
            {"role": "assistant", "content": "Telescopes collect and focus light from distant objects. Optical telescopes use lenses (refracting) or mirrors (reflecting) to gather light and magnify images."},
        ], "followup": "What makes the James Webb Space Telescope special?"},
        {"context": [
            {"role": "user", "content": "What is the periodic table?"},
            {"role": "assistant", "content": "The periodic table organizes all known chemical elements by their atomic number, electron configuration, and chemical properties into rows (periods) and columns (groups)."},
        ], "followup": "Why are noble gases unreactive?"},
        {"context": [
            {"role": "user", "content": "How does encryption work?"},
            {"role": "assistant", "content": "Encryption converts readable data into an unreadable format using mathematical algorithms and keys. Only someone with the correct key can decrypt and read the original data."},
        ], "followup": "What is the difference between symmetric and asymmetric encryption?"},
        {"context": [
            {"role": "user", "content": "What is supply and demand?"},
            {"role": "assistant", "content": "Supply and demand is an economic model explaining price determination. When demand exceeds supply, prices rise; when supply exceeds demand, prices fall."},
        ], "followup": "What happens when the government sets price controls?"},
        {"context": [
            {"role": "user", "content": "How do hurricanes form?"},
            {"role": "assistant", "content": "Hurricanes form over warm ocean water when moist air rises, creating an area of low pressure. As more warm air rushes in and rises, it begins to rotate due to the Coriolis effect."},
        ], "followup": "Why are hurricanes getting stronger in recent years?"},
    ]

    samples = []
    for i in range(n):
        scenario = scenarios[i % len(scenarios)]
        msgs = [{"role": "system", "content": "You are a helpful and knowledgeable assistant."}]
        msgs.extend(scenario["context"])
        msgs.append({"role": "user", "content": scenario["followup"]})
        samples.append({
            "messages": msgs,
            "task_description": f"Follow-up question: {scenario['followup']}",
            "category": "multi_turn",
            "id": f"mt_{i+1:03d}",
        })
    return samples


def generate_all_samples():
    """Generate all 450 benchmark samples."""
    random.seed(42)
    return {
        "instruction_following": _gen_instruction_following(100),
        "knowledge_qa": _gen_knowledge_qa(200),
        "code_generation": _gen_code_generation(50),
        "summarization": _gen_summarization(50),
        "multi_turn": _gen_multi_turn(50),
    }


# =====================================================================
# Type ID Assignment for Multi-turn
# =====================================================================

def assign_type_ids_for_messages(tokenizer, messages):
    """Assign type_ids for arbitrary message sequences.

    system -> 0, user -> 1, assistant -> 0
    Returns numpy array of type_ids matching the tokenized sequence.
    """
    type_ids = []
    prev_len = 0

    for i in range(len(messages)):
        partial = messages[:i + 1]
        try:
            ids = tokenizer.apply_chat_template(
                partial, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                partial, add_generation_prompt=False,
            )
        cur_len = len(ids)
        role = messages[i]["role"]
        type_id = 1 if role == "user" else 0
        type_ids.extend([type_id] * (cur_len - prev_len))
        prev_len = cur_len

    # Add generation prompt tokens (assistant type)
    try:
        full_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        full_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
        )
    if len(full_ids) > prev_len:
        type_ids.extend([0] * (len(full_ids) - prev_len))

    return np.array(type_ids)


# =====================================================================
# Response Generation
# =====================================================================

def strip_thinking(text):
    """Remove Qwen3's <think>...</think> blocks from response."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def generate_response(model, tokenizer, messages, max_new_tokens=512):
    """Generate a response from the model given messages."""
    import torch

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return strip_thinking(response)


# =====================================================================
# LLM-as-Judge
# =====================================================================

JUDGE_SYSTEM = (
    "You are a fair evaluator. Score the assistant's response on a scale of 1 to 5.\n"
    "5 = Excellent: Complete, accurate, well-written\n"
    "4 = Good: Mostly correct with minor issues\n"
    "3 = Adequate: Partially correct or incomplete\n"
    "2 = Poor: Significant errors or missing key information\n"
    "1 = Very poor: Incorrect, irrelevant, or unhelpful\n"
    "Output ONLY a single number from 1 to 5."
)


def judge_response(model, tokenizer, task_description, response):
    """Score a response 1-5 using LLM-as-judge."""
    import torch

    judge_messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": (
            f"Task: {task_description}\n\n"
            f"Response: {response}\n\n"
            f"Score (1-5):"
        )},
    ]

    try:
        text = tokenizer.apply_chat_template(
            judge_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            judge_messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    judge_output = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    judge_output = strip_thinking(judge_output)

    # Parse score: find first digit 1-5
    for ch in judge_output:
        if ch in "12345":
            return int(ch)
    return 3  # default


# =====================================================================
# Evaluation Pipeline
# =====================================================================

def evaluate_condition(model, tokenizer, samples, hooks=None, condition_name="baseline"):
    """Generate responses for all samples under a condition.

    Returns dict[category] -> list of response strings.
    """
    import torch

    all_responses = {}
    total = sum(len(v) for v in samples.values())
    count = 0

    for category, sample_list in samples.items():
        print(f"\n  [{condition_name}] {category} ({len(sample_list)} samples)...")
        responses = []
        for i, sample in enumerate(sample_list):
            count += 1
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    {i+1}/{len(sample_list)} (total {count}/{total})", flush=True)

            messages = sample["messages"]

            # Set type_ids for rotation defense
            if hooks is not None:
                type_ids = assign_type_ids_for_messages(tokenizer, messages)
                type_ids_tensor = torch.tensor([type_ids], dtype=torch.long).to(
                    next(model.parameters()).device
                )
                hooks.set_type_ids(type_ids_tensor)

            response = generate_response(model, tokenizer, messages)
            responses.append(response)

        all_responses[category] = responses
    return all_responses


def judge_all_responses(model, tokenizer, samples, responses):
    """Score all responses using LLM-as-judge.

    Returns dict[category] -> list of int scores (1-5).
    """
    all_scores = {}
    total = sum(len(v) for v in samples.values())
    count = 0

    for category, sample_list in samples.items():
        print(f"\n  [judging] {category} ({len(sample_list)} samples)...")
        scores = []
        for i, sample in enumerate(sample_list):
            count += 1
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    {i+1}/{len(sample_list)} (total {count}/{total})", flush=True)

            score = judge_response(
                model, tokenizer,
                sample["task_description"],
                responses[category][i],
            )
            scores.append(score)

        all_scores[category] = scores
    return all_scores


# =====================================================================
# Statistics
# =====================================================================

def compute_statistics(baseline_scores, defense_scores):
    """Compute per-category statistics with paired t-test and 95% CI."""
    results = {}
    all_bl = []
    all_df = []

    for category in baseline_scores:
        bl = np.array(baseline_scores[category], dtype=float)
        df = np.array(defense_scores[category], dtype=float)
        all_bl.extend(bl)
        all_df.extend(df)

        diff = df - bl
        mean_diff = np.mean(diff)
        se = np.std(diff, ddof=1) / np.sqrt(len(diff))
        ci_lower = mean_diff - 1.96 * se
        ci_upper = mean_diff + 1.96 * se

        t_stat, p_value = stats.ttest_rel(df, bl)

        results[category] = {
            "n_samples": len(bl),
            "baseline_mean": float(np.mean(bl)),
            "baseline_std": float(np.std(bl, ddof=1)),
            "defense_mean": float(np.mean(df)),
            "defense_std": float(np.std(df, ddof=1)),
            "mean_diff": float(mean_diff),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_degradation": bool(p_value < 0.05 and mean_diff < 0),
        }

    # Overall aggregate
    all_bl = np.array(all_bl, dtype=float)
    all_df = np.array(all_df, dtype=float)
    diff = all_df - all_bl
    mean_diff = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    t_stat, p_value = stats.ttest_rel(all_df, all_bl)

    results["overall"] = {
        "n_samples": len(all_bl),
        "baseline_mean": float(np.mean(all_bl)),
        "defense_mean": float(np.mean(all_df)),
        "mean_diff": float(mean_diff),
        "ci_95_lower": float(mean_diff - 1.96 * se),
        "ci_95_upper": float(mean_diff + 1.96 * se),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_degradation": bool(p_value < 0.05 and mean_diff < 0),
    }

    return results


# =====================================================================
# Figure Generation
# =====================================================================

def generate_figure(stats_results, output_dir=OUTPUT_DIR):
    """Generate bar chart comparing baseline vs defense scores per category."""
    os.makedirs(output_dir, exist_ok=True)

    categories = [c for c in stats_results if c != "overall"]
    cat_labels = [c.replace("_", "\n") for c in categories]

    bl_means = [stats_results[c]["baseline_mean"] for c in categories]
    df_means = [stats_results[c]["defense_mean"] for c in categories]
    bl_stds = [stats_results[c]["baseline_std"] for c in categories]
    df_stds = [stats_results[c]["defense_std"] for c in categories]
    p_values = [stats_results[c]["p_value"] for c in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("D1: Benign Performance Benchmark — Baseline vs LoRA+Rotation",
                 fontsize=13, fontweight="bold")

    # Panel 1: Mean scores
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, bl_means, width, label="Baseline",
                    color="steelblue", edgecolor="darkblue", alpha=0.85,
                    yerr=[s / np.sqrt(stats_results[c]["n_samples"])
                          for s, c in zip(bl_stds, categories)],
                    capsize=3)
    bars2 = ax1.bar(x + width / 2, df_means, width, label="LoRA + Rotation",
                    color="coral", edgecolor="darkred", alpha=0.85,
                    yerr=[s / np.sqrt(stats_results[c]["n_samples"])
                          for s, c in zip(df_stds, categories)],
                    capsize=3)
    ax1.set_ylabel("Mean Score (1-5)")
    ax1.set_title("(a) Per-Category Mean Scores")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels, fontsize=9)
    ax1.set_ylim(0, 5.5)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add significance markers
    for i, p in enumerate(p_values):
        if p < 0.01:
            marker = "**"
        elif p < 0.05:
            marker = "*"
        else:
            marker = "n.s."
        y_max = max(bl_means[i], df_means[i]) + 0.3
        ax1.text(x[i], y_max, marker, ha="center", va="bottom", fontsize=10,
                 color="green" if marker == "n.s." else "red")

    # Panel 2: Score differences with 95% CI
    diffs = [stats_results[c]["mean_diff"] for c in categories]
    ci_lo = [stats_results[c]["ci_95_lower"] for c in categories]
    ci_hi = [stats_results[c]["ci_95_upper"] for c in categories]
    errors = [[d - lo for d, lo in zip(diffs, ci_lo)],
              [hi - d for d, hi in zip(diffs, ci_hi)]]

    colors = ["green" if lo <= 0 <= hi else "red"
              for lo, hi in zip(ci_lo, ci_hi)]
    ax2.barh(x, diffs, xerr=errors, height=0.6, color=colors, alpha=0.7,
             edgecolor="gray", capsize=4)
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Score Difference (Defense - Baseline)")
    ax2.set_title("(b) Score Differences with 95% CI")
    ax2.set_yticks(x)
    ax2.set_yticklabels(cat_labels, fontsize=9)
    ax2.grid(True, alpha=0.3, axis="x")

    # Annotate p-values
    for i, (p, d) in enumerate(zip(p_values, diffs)):
        ax2.text(max(ci_hi[i], 0) + 0.05, i, f"p={p:.3f}", va="center",
                 fontsize=8, color="gray")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "fig_benign_eval.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {fig_path}")
    return fig_path


# =====================================================================
# Real Model Evaluation
# =====================================================================

def run_real(args):
    """Run full evaluation with real model on GPU."""
    import torch
    import yaml
    from peft import PeftModel
    from model.hook_attention import TypedRoPEHooks
    from utils import load_model_from_config

    # Load experiment config
    exp_config_path = os.path.join(CONFIG_DIR, "experiment.yaml")
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)
    target_subspaces = exp_config["target_subspaces"]
    rotation_angle = exp_config["max_safe_angle"]

    # Determine model config from adapter
    config_path = args.config
    if config_path is None:
        adapter_cfg_path = os.path.join(args.adapter_dir, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            with open(adapter_cfg_path) as f:
                acfg = json.load(f)
            base_model = acfg.get("base_model_name_or_path", "")
            if "qwen" in base_model.lower():
                config_path = os.path.join(CONFIG_DIR, "qwen3_8b_train.yaml")
            else:
                config_path = os.path.join(CONFIG_DIR, "llama8b.yaml")
        else:
            config_path = os.path.join(CONFIG_DIR, "llama8b.yaml")
    print(f"Model config: {config_path}")
    print(f"LoRA adapter: {args.adapter_dir}")
    print(f"Target subspaces: {target_subspaces}")
    print(f"Rotation angle: {rotation_angle}")

    # Generate benchmark samples
    samples = generate_all_samples()
    total = sum(len(v) for v in samples.values())
    print(f"\nGenerated {total} benchmark samples")

    # Load model
    print("\nLoading model...")
    device_map = f"cuda:{args.gpu}" if args.gpu is not None else "auto"
    model, tokenizer, _ = load_model_from_config(config_path, device_map=device_map)
    model.eval()

    # === Condition 1: Baseline (no LoRA, no rotation) ===
    print("\n" + "=" * 60)
    print("CONDITION 1: Baseline (no LoRA, no rotation)")
    print("=" * 60)
    t0 = time.time()
    baseline_responses = evaluate_condition(model, tokenizer, samples,
                                            condition_name="baseline")
    t_baseline = time.time() - t0
    print(f"  Baseline generation: {t_baseline:.1f}s")

    # === Condition 2: LoRA + Rotation ===
    print("\n" + "=" * 60)
    print("CONDITION 2: LoRA + Rotation")
    print("=" * 60)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    # Install rotation hooks
    hooks = TypedRoPEHooks(target_subspaces, rotation_angle)
    hooks.install(model)

    t0 = time.time()
    defense_responses = evaluate_condition(model, tokenizer, samples,
                                           hooks=hooks, condition_name="defense")
    t_defense = time.time() - t0
    print(f"  Defense generation: {t_defense:.1f}s")

    # === Cleanup for judging ===
    hooks.remove()
    # Disable LoRA adapter layers so base model is used for judging
    model.disable_adapter_layers()

    # === Judging ===
    print("\n" + "=" * 60)
    print("JUDGING: Scoring responses with base model")
    print("=" * 60)

    print("\n  Judging baseline responses...")
    t0 = time.time()
    baseline_scores = judge_all_responses(model, tokenizer, samples, baseline_responses)
    t_judge_bl = time.time() - t0

    print("\n  Judging defense responses...")
    t0 = time.time()
    defense_scores = judge_all_responses(model, tokenizer, samples, defense_responses)
    t_judge_df = time.time() - t0

    print(f"\n  Judging time: baseline={t_judge_bl:.1f}s, defense={t_judge_df:.1f}s")

    # === Statistics ===
    stats_results = compute_statistics(baseline_scores, defense_scores)

    # === Print Summary ===
    print("\n" + "=" * 60)
    print("D1: Benign Performance Benchmark Results")
    print("=" * 60)

    for cat in [c for c in stats_results if c != "overall"]:
        r = stats_results[cat]
        sig = "DEGRADED" if r["significant_degradation"] else "OK"
        print(f"  {cat:25s}: baseline={r['baseline_mean']:.2f}, "
              f"defense={r['defense_mean']:.2f}, "
              f"diff={r['mean_diff']:+.3f}, p={r['p_value']:.4f} [{sig}]")

    overall = stats_results["overall"]
    print(f"\n  {'overall':25s}: baseline={overall['baseline_mean']:.2f}, "
          f"defense={overall['defense_mean']:.2f}, "
          f"diff={overall['mean_diff']:+.3f}, p={overall['p_value']:.4f}")

    # Verification
    any_degradation = any(
        stats_results[c]["significant_degradation"]
        for c in stats_results if c != "overall"
    )
    overall_degradation = overall["significant_degradation"]
    verification = not any_degradation and not overall_degradation

    print(f"\n  Any category with significant degradation: {any_degradation}")
    print(f"  Overall significant degradation: {overall_degradation}")
    print(f"\n  Verification (no significant degradation): "
          f"{'PASS' if verification else 'FAIL'}")

    # === Save Results ===
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "mode": "real",
        "model_config": config_path,
        "adapter_dir": args.adapter_dir,
        "target_subspaces": target_subspaces,
        "rotation_angle": rotation_angle,
        "n_total_samples": total,
        "categories": {
            cat: {"n_samples": len(samples[cat])}
            for cat in samples
        },
        "timing": {
            "baseline_generation_s": t_baseline,
            "defense_generation_s": t_defense,
            "baseline_judging_s": t_judge_bl,
            "defense_judging_s": t_judge_df,
        },
        "baseline_scores": baseline_scores,
        "defense_scores": defense_scores,
        "statistics": stats_results,
        "verification": {
            "any_category_degradation": any_degradation,
            "overall_degradation": overall_degradation,
            "pass": verification,
        },
    }

    # Save sample responses (first 5 per category for inspection)
    results["sample_responses"] = {}
    for cat in samples:
        results["sample_responses"][cat] = []
        for i in range(min(5, len(samples[cat]))):
            results["sample_responses"][cat].append({
                "id": samples[cat][i]["id"],
                "task": samples[cat][i]["task_description"],
                "baseline_response": baseline_responses[cat][i][:500],
                "defense_response": defense_responses[cat][i][:500],
                "baseline_score": baseline_scores[cat][i],
                "defense_score": defense_scores[cat][i],
            })

    results_path = os.path.join(args.output_dir, "benign_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    generate_figure(stats_results, args.output_dir)

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="D1: Benign Performance Benchmark")
    parser.add_argument("--adapter-dir", default="outputs/lora_adapter",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--config", default=None,
                        help="Model config YAML (auto-detected from adapter if not set)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory")
    args = parser.parse_args()

    run_real(args)


if __name__ == "__main__":
    main()
