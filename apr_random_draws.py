# Copyright (C) [2024-2026] APR PRAVEENS LUXURIA SPORTS COMMITTEE
# Copyright (C) APR PRAVEENS LUXURIA ASSOCIATION 
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#Dedicated to Suryakumar Yadav (SKY), who patiently waited for extended years to get his chance,
#then seized the moment and revolutionized the way T20 cricket is played.

#Description:
#This script processes sports enrollment data from a CSV file framed by the APR Sports Committee in late 2024. 
#It provides options to:
#- View the list of sports, participants, add/delete participants on the fly.
#- Create random pairs and schedules for selected sports.
#- Generate all possible sports pairs and schedules, which are outputted into timestamped folders.
#
#Scheduling Options:
#- For groups of 3, 4, 5, and 6 participants: Round-robin scheduling is used.
#- For groups larger than 6: Knockout scheduling is applied.
#
#Features:
#- Parses input CSV files containing enrollment data.
#- Supports interactive selection of sports for creating random draws and schedules.
#- Automatically generates timestamped folders containing outputs for all sports.
#
#Dependencies:
#- Python 3.7 or above
#- pandas (install using `pip install pandas`)
#
#Installation:
#Ensure you have Python installed on your system. Install the required library using:
#pip install pandas

import pandas as pd
import random
import re
import os
import sys
from datetime import datetime


def load_data():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("Error: No file path provided.")
        print("Usage: python apr_random_draws_v3.py <path_to_csv_file>")
        sys.exit(1)

    df = pd.read_csv(file_path)
    
    # Identify the correct sport column
    if 'Please select the sport in which you wish to participate:' in df.columns:
        sport_column = 'Please select the sport in which you wish to participate:'
    elif 'Sport you wish to participate in:' in df.columns:
        sport_column = 'Sport you wish to participate in:'
    else:
        raise KeyError("Sport selection column not found. Please check the file format.")
    
    # Check for the Age column
    age_column_present = 'Age' in df.columns
    if age_column_present:
        def parse_age_range(age_str):
            match = re.match(r"(\d+)-(\d+) years", age_str)
            if match:
                min_age, max_age = int(match.group(1)), int(match.group(2))
                return min_age, max_age
            return None, None

        df['Min_Age'], df['Max_Age'] = zip(*df['Age'].apply(lambda x: parse_age_range(str(x))))

    # Handle sports splitting based on ";" or "," delimiters
    if sport_column + '.1' in df.columns:
        df['Sports'] = df[sport_column].fillna('') + ';' + df[sport_column + '.1'].fillna('')
    else:
        df['Sports'] = df[sport_column].fillna('')
    
    # Split sports by ";" or "," and create an expanded DataFrame
    df['Sports'] = df['Sports'].apply(lambda x: [sport.strip() for sport in re.split(r'[;,]', x) if sport.strip()])
    df_expanded = df.explode('Sports').dropna(subset=['Sports'])
    return df_expanded, age_column_present


def create_random_pairs(data, seed, group_size=2, sport=None):
    random.seed(seed)
    shuffled_data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # For chess, treat each participant as a single entry
    if sport == "Chess":
        groups = [[participant] for _, participant in shuffled_data.iterrows()]
    else:
        # Standard pairing logic for other sports
        if len(shuffled_data) % group_size != 0:
            bye_player = shuffled_data.iloc[-1]
            shuffled_data = shuffled_data.iloc[:-1]
            groups = [shuffled_data[i:i + group_size].values.tolist() for i in range(0, len(shuffled_data), group_size)]
            groups.append([[bye_player['Name (FirstName LastName)'], bye_player['Phase'], bye_player['Villa Number'], bye_player['Phone Number'], "Bye"]])
        else:
            groups = [shuffled_data[i:i + group_size].values.tolist() for i in range(0, len(shuffled_data), group_size)]

    return groups


def save_pairs_to_csv(folder_name, sport, gender, age_range, pairs):
    age_suffix = f"_{age_range[0]}-{age_range[1]}y" if age_range else ""
    file_name = f"{sport}_{gender if gender else 'all'}{age_suffix}.csv"
    file_path = os.path.join(folder_name, file_name)

    data = []
    for pair in pairs:
        if len(pair) == 2:
            data.append([pair[0][0], pair[0][1], pair[0][2], pair[0][3], pair[1][0], pair[1][1], pair[1][2], pair[1][3]])
        else:
            data.append([pair[0][0], pair[0][1], pair[0][2], pair[0][3], "Bye", "", "", ""])

    df = pd.DataFrame(data, columns=[
        'Participant 1 Name', 'Participant 1 Phase', 'Participant 1 Villa', 'Participant 1 Phone',
        'Participant 2 Name', 'Participant 2 Phase', 'Participant 2 Villa', 'Participant 2 Phone'
    ])
    df.to_csv(file_path, index=False)
    print(f"Pairs saved to {file_path}")


def generate_all_combinations(df, age_column_present, stored_draws, seed, sport=None):
    """Generates all combinations for the specified sport, gender, and age range, and saves results to CSV using the provided seed."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"APR_RANDOM_DRAWS_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # If no sport is specified, use all sports
    sports = [sport] if sport else ["Badminton", "Table Tennis", "Tennis", "Chess", "Carroms"]
    genders = ["Male", "Female"]
    age_ranges = [(8, 10), (11, 12), (13, 16)] if age_column_present else [None]

    for selected_sport in sports:
        for gender in genders:
            for age_range in age_ranges:
                participants_df = df[df['Sports'] == selected_sport]
                participants_df = participants_df[participants_df['Gender'].str.lower() == gender.lower()]

                if age_range:
                    min_age, max_age = age_range
                    participants_df = participants_df[(participants_df['Min_Age'].notna()) &
                                                      (participants_df['Min_Age'] <= max_age) &
                                                      (participants_df['Max_Age'] >= min_age)]

                if participants_df.empty:
                    print(f"No participants found for {selected_sport}, {gender}, Age Range {age_range}.")
                    continue

                # Generate random pairs with the provided seed
                pairs = create_random_pairs(participants_df[['Name (FirstName LastName)', 'Phase', 'Villa Number', 'Phone Number']], seed, sport=selected_sport)

                # Remove "Bye" for single participants in the pairs
                cleaned_pairs = [pair for pair in pairs if len(pair) > 1 or (len(pair) == 1 and pair[0][0] != "Bye")]

                # Display and save each round on the console
                print(f"\n--- {selected_sport} ({gender}) Age Range: {age_range if age_range else 'All Ages'} ---")
                handle_knockout_tournament(cleaned_pairs, folder_name, selected_sport, gender, participants_df, age_range)


def display_menu():
    draw_sports = ["Badminton", "Table Tennis", "Tennis", "Chess", "Carroms"]
    print("\nSelect a Sport for Draws:")
    for i, sport in enumerate(draw_sports, start=1):
        print(f"{i}. {sport}")
    print("0. Exit")
    print("6. Generate all categories and save")
    return draw_sports

def list_draws(draws):
    if not draws:
        print("\nNo draws available yet.")
        return
    
    print("\nExisting Draws:")
    for i, (sport, gender, groups) in enumerate(draws, start=1):
        print(f"\nDraw {i} - Sport: {sport}, Gender: {gender if gender else 'All'}")
        for group in groups:
            if len(group) == 2:
                print(f"  Pair: {group[0][0]} (Phase {group[0][1]}, Villa {group[0][2]}) and {group[1][0]} (Phase {group[1][1]}, Villa {group[1][2]})")
            else:
                print(f"  Single with Bye: {group[0][0]} (Phase {group[0][1]}, Villa {group[0][2]})")

def add_participant(df, sport):
    name = input("Enter participant's full name: ")
    phase = input("Enter participant's phase: ")
    villa = input("Enter participant's villa number: ")
    phone = input("Enter participant's phone number: ")
    gender = input("Enter participant's gender (Male/Female): ")

    new_entry = pd.DataFrame({
        'Name (FirstName LastName)': [name],
        'Phase': [phase],
        'Villa Number': [villa],
        'Phone Number': [phone],
        'Gender': [gender],
        'Sports': [sport]
    })
    
    # Use pd.concat to add the new entry
    return pd.concat([df, new_entry], ignore_index=True)

def delete_participant(df, sport, gender):
    print("\nCurrent Participants:")
    list_participants(df, sport, gender)
    index = int(input("Enter the participant number to delete: ")) - 1
    return df.drop(df.index[index])

def list_participants(df, sport, gender=None):
    participants = df[df['Sports'] == sport]
    if gender:
        participants = participants[participants['Gender'].str.lower() == gender.lower()]
    
    if participants.empty:
        print(f"No participants found for {sport} ({gender if gender else 'All'}).")
    else:
        print(f"\nParticipants in {sport} ({gender if gender else 'All'}):")
        for i, row in enumerate(participants[['Name (FirstName LastName)', 'Phase', 'Villa Number', 'Phone Number']].values, start=1):
            print(f"{i}. {row[0]} (Phase {row[1]}, Villa {row[2]}, Phone {row[3]})")

def get_match_label(round_number, match_number):
    """Generates a label for each match winner based on the round and match number."""
    return f"Winner of Round {round_number}, Match {match_number}"

def label_teams(pairs):
    """Assigns team names sequentially without adding 'Bye' as a team member."""
    teams = {f"Team {i+1}": pair for i, pair in enumerate(pairs)}
    print("\nTeams:")
    for team, members in teams.items():
        member_names = " and ".join(member[0] for member in members)  # Combine names for each team without 'Bye'
        print(f"{team}: {member_names}")
    return teams


def arrange_knockout_with_labels(teams):
    """Organizes knockout rounds with labels and adds byes only in the schedule."""
    rounds = []
    current_teams = list(teams.keys())
    round_number = 1

    while len(current_teams) > 1:
        round_matches = []
        match_labels = []
        random.shuffle(current_teams)

        # If there's an odd number of teams, give a bye to one team for the next round
        if len(current_teams) % 2 != 0:
            bye_team = current_teams.pop()  # Remove and get the team with a bye
            match_labels.append(bye_team)
            print(f"{bye_team} receives a bye to the next round")

        # Pair up the remaining teams
        for i in range(0, len(current_teams), 2):
            team1, team2 = current_teams[i], current_teams[i + 1]
            match_label = get_match_label(round_number, (i // 2) + 1)
            round_matches.append((team1, team2))
            match_labels.append(match_label)

        rounds.append((round_number, round_matches, match_labels))
        
        # Move winners to the next round (simulate by advancing the labels)
        current_teams = match_labels
        round_number += 1

    return rounds


def save_teams_and_schedule_to_csv(folder_name, sport, teams, rounds):
    """Saves teams and knockout schedule to a single CSV with formatting for readability."""
    file_name = f"{sport}_Tournament_Schedule.csv"
    file_path = os.path.join(folder_name, file_name)
    
    data = []
    
    # Add Teams Section
    data.append(["Teams", "", ""])
    for team_name, members in teams.items():
        member_names = " and ".join(member[0] for member in members)  # Combine names for each team
        data.append([team_name, member_names, ""])
    data.append(["", "", ""])  # Blank row for spacing after Teams section
    
    # Add Knockout Schedule Section
    data.append(["Knockout Schedule", "", ""])
    for round_number, matches, _ in rounds:
        # Determine if the round is a semifinal or final based on match count
        round_label = f"Round {round_number}"
        if len(matches) == 2:
            round_label = "Semifinal"
        elif len(matches) == 1:
            round_label = "Final"
        
        # Add round label as a header row with a blank row for spacing
        data.append(["", "", ""])  # Blank row for spacing before each round
        data.append([round_label, "", ""])

        # Add each match in the round
        for match in matches:
            team1, team2 = match
            data.append(["", team1, team2])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=["Section", "Team 1", "Team 2"])
    df.to_csv(file_path, index=False)
    print(f"Tournament schedule saved to {file_path}")


def save_knockout_schedule_to_csv(folder_name, sport, rounds):
    """Saves knockout round schedule with labeled rounds and spacing to CSV."""
    file_name = f"{sport}_Knockout_Schedule.csv"
    file_path = os.path.join(folder_name, file_name)
    
    data = []
    for round_number, matches, _ in rounds:
        # Determine if the round is a semifinal or final based on match count
        round_label = f"Round {round_number}"
        if len(matches) == 2:
            round_label = "Semifinal"
        elif len(matches) == 1:
            round_label = "Final"
        
        # Add round label as a header row with a blank row for spacing
        data.append([round_label, "", ""])
        data.append(["", "", ""])  # Blank row for spacing

        # Add each match in the round
        for match in matches:
            team1, team2 = match
            data.append([round_label, team1, team2])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=["Round", "Team 1", "Team 2"])
    df.to_csv(file_path, index=False)
    print(f"Knockout schedule saved to {file_path}")



def save_full_schedule_to_csv(folder_name, file_name, participants, teams, rounds,  age_range=None):
    """Saves participants, teams, and schedule to a single CSV with age range in the filename if applicable."""
    # Ensure age range is only appended once to the file name
    if '_schedule' not in file_name:
        file_name += '_schedule'
    file_path = os.path.join(folder_name, f"{file_name}.csv")

    data = []

    # Add Participants Section
    data.append(["Participants", "", "", "", ""])
    data.append(["Name", "Phase", "Villa Number", "Phone Number", ""])
    for participant in participants:
        name, phase, villa, phone = participant
        data.append([name, phase, villa, phone, ""])
    data.append(["", "", "", "", ""])  # Blank row for spacing after Participants section

    # Add Teams Section
    data.append(["Teams", "", ""])
    for team_name, members in teams.items():
        member_names = " and ".join(member[0] for member in members)  # Combine names for each team
        data.append([team_name, member_names, ""])
    data.append(["", "", ""])  # Blank row for spacing after Teams section

    # Add Tournament Schedule Section
    last_round_label = None
    match_counter = 1
    for round_label, matches, _ in rounds:
        # Add round label only once for round-robin format
        if round_label != last_round_label:
            data.append(["", "", ""])  # Blank row for spacing before each round
            data.append([round_label, "", ""])
            match_counter = 1  # Reset match number for each new round label

        # Add each match with numbering
        for team1, team2 in matches:
            match_label = f"Match {match_counter}"
            data.append([match_label, team1, team2])
            match_counter += 1

        last_round_label = round_label  # Track the last used round label to avoid repetition for round robin

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=["Section", "Team 1", "Team 2", "Team Details", "Extra"])
    df.to_csv(file_path, index=False)
    print(f"Tournament schedule saved to {file_path}")

def handle_knockout_tournament(pairs, folder_name, sport, gender, participants_df, age_range=None):
    """Handles the tournament structure and ensures each schedule is saved with an age-specific filename."""
    teams = label_teams(pairs)  # Generate and display team labels
    team_count = len(teams)
    
    rounds = []  # Initialize rounds to hold all matches and rounds
    final_match = None

    # Format the file name to include age range if provided
    age_suffix = f"_{age_range[0]}-{age_range[1]}" if age_range else ""
    file_name = f"{sport}_{gender}{age_suffix}_schedule"

    if team_count == 2:
        # Direct final for 2 teams
        print("\n--- Final Match ---")
        final_match = ("Final", [(list(teams.keys())[0], list(teams.keys())[1])], None)
        print(f"Final Match: {list(teams.keys())[0]} vs {list(teams.keys())[1]}")

    elif team_count in [3, 4]:
        print(f"\n--- Round Robin for {team_count} Teams ---")

        # Round-robin schedule for 3 or 4 teams
        match_counter = 1
        for i, team1 in enumerate(teams.keys()):
            for j, team2 in enumerate(teams.keys()):
                if i < j:
                    rounds.append((f"Round Robin", [(team1, team2)], None))
                    print(f"Match {match_counter}: {team1} vs {team2}")
                    match_counter += 1

        # Final match between top 2 teams from round robin
        print("\n--- Final Match ---")
        final_match = ("Final", [("Top 1 Team", "Top 2 Team")], None)
        print("Final Match: Top 1 Team vs Top 2 Team")

    elif team_count == 5:
        print("\n--- Round Robin for 5 Teams ---")
        
        # Create a round-robin schedule for all 5 teams
        match_counter = 1
        for i, team1 in enumerate(teams.keys()):
            for j, team2 in enumerate(teams.keys()):
                if i < j:
                    rounds.append((f"Round Robin", [(team1, team2)], None))
                    print(f"Match {match_counter}: {team1} vs {team2}")
                    match_counter += 1

        # Top 4 teams move to semifinals
        print("\n--- Semifinals ---")
        rounds.append(("Semifinal", [("Top 1 Team", "Top 4 Team"), ("Top 2 Team", "Top 3 Team")], None))
        print("Semifinal Match 1: Top 1 Team vs Top 4 Team")
        print("Semifinal Match 2: Top 2 Team vs Top 3 Team")

        # Final match between winners of semifinals
        print("\n--- Final Match ---")
        final_match = ("Final", [("Winner of Semifinal 1", "Winner of Semifinal 2")], None)
        print("Final Match: Winner of Semifinal 1 vs Winner of Semifinal 2")
        
    elif team_count == 6:
        print("\n--- Group Stage for 6 Teams ---")
        group1, group2 = list(teams.keys())[:3], list(teams.keys())[3:]
        
        # Round-robin within Group 1
        print("\nGroup 1:")
        match_counter = 1
        for i in range(len(group1)):
            for j in range(i + 1, len(group1)):
                team1, team2 = group1[i], group1[j]
                rounds.append((f"Group 1 Round Robin", [(team1, team2)], None))
                print(f"Match {match_counter}: {team1} vs {team2}")
                match_counter += 1

        # Round-robin within Group 2
        print("\nGroup 2:")
        match_counter = 1
        for i in range(len(group2)):
            for j in range(i + 1, len(group2)):
                team1, team2 = group2[i], group2[j]
                rounds.append((f"Group 2 Round Robin", [(team1, team2)], None))
                print(f"Match {match_counter}: {team1} vs {team2}")
                match_counter += 1
        
        # Top 2 teams from each group advance to the semifinals
        print("\n--- Semifinals ---")
        rounds.append(("Semifinal", [("Top Team Group 1", "Second Team Group 2"), ("Top Team Group 2", "Second Team Group 1")], None))
        print("Semifinal Match 1: Top Team Group 1 vs Second Team Group 2")
        print("Semifinal Match 2: Top Team Group 2 vs Second Team Group 1")

        # Final match between winners of semifinals
        print("\n--- Final Match ---")
        final_match = ("Final", [("Winner of Semifinal 1", "Winner of Semifinal 2")], None)
        print("Final Match: Winner of Semifinal 1 vs Winner of Semifinal 2")
        
    else:
        # Standard knockout for other team counts
        rounds = arrange_knockout_with_labels(teams)
        
        # Display each round on the console
        for round_number, matches, _ in rounds:
            print(f"\nRound {round_number}:")
            for match_index, (team1, team2) in enumerate(matches, start=1):
                print(f"  Match {match_index}: {team1} vs {team2}")

    # Add final match to rounds if applicable
    if final_match:
        rounds.append(final_match)

    # Gather participants' details
    participants = participants_df[['Name (FirstName LastName)', 'Phase', 'Villa Number', 'Phone Number']].values.tolist()
    
    # Save participants, teams, and full schedule (including round-robin and final matches) to a single CSV file with age suffix
    save_full_schedule_to_csv(folder_name, file_name, participants, teams, rounds)


def main():
    try:
        # Load the data and prepare necessary variables
        df_expanded, age_column_present = load_data()
        draws = []

        while True:
            draw_sports = display_menu()
            choice = input("\nEnter the sport number (or 0 to exit): ")

            if choice == '0':
                print("Exiting the program.")
                break

            elif choice == '6':
                seed = int(input("Enter a random seed for the draw (a number): "))
                generate_all_combinations(df_expanded, age_column_present, draws, seed)
                print("All categories generated and saved.")

            elif choice.isdigit() and 1 <= int(choice) <= len(draw_sports):
                sport = draw_sports[int(choice) - 1]
                gender = input("Do you want to filter by gender? (Enter 'Male', 'Female', or leave blank for all): ").strip()

                # Age-based selection menu if age data is available
                age_range = None
                if age_column_present:
                    print("\nSelect Age Range:")
                    print("1. Ages 8-10")
                    print("2. Ages 11-12")
                    print("3. Ages 13-16")
                    print("4. All Ages")
                    age_choice = input("Enter your choice: ")
                    if age_choice == '1':
                        age_range = (8, 10)
                    elif age_choice == '2':
                        age_range = (11, 12)
                    elif age_choice == '3':
                        age_range = (13, 16)
                    elif age_choice == '4':
                        age_range = None

                # Filter participants based on sport, gender, and age range
                participants_df = df_expanded[df_expanded['Sports'] == sport]
                if gender:
                    participants_df = participants_df[participants_df['Gender'].str.lower() == gender.lower()]
                if age_range:
                    min_age, max_age = age_range
                    participants_df = participants_df[(participants_df['Min_Age'].notna()) &
                                                      (participants_df['Min_Age'] <= max_age) &
                                                      (participants_df['Max_Age'] >= min_age)]

                if participants_df.empty:
                    print(f"No participants found for {sport}, {gender if gender else 'All'}, Age Range: {age_range}.")
                    continue

                while True:
                    # Interactive menu for each selected sport, gender, and age range
                    print("\nOptions:")
                    print("1. List participants")
                    print("2. Draw random pairs")
                    print("3. List existing draws")
                    print("4. Add participant")
                    print("5. Delete participant")
                    print("6. Back to main menu")
                    action = input("Choose an option: ").strip()

                    if action == '1':
                        # List participants
                        list_participants(participants_df, sport, gender)

                    elif action == '2':
                        # Draw random pairs
                        seed = int(input("Enter a random seed for the draw (a number): "))
                        # If the selected sport is "Chess," treat participants as individuals rather than pairs
                        if sport.lower() == "chess":
                            pairs = [[participant] for participant in participants_df[['Name (FirstName LastName)', 'Phase', 'Villa Number', 'Phone Number']].values]
                        else:
                            pairs = create_random_pairs(participants_df[['Name (FirstName LastName)', 'Phase', 'Villa Number', 'Phone Number']], seed)
                        draws.append((sport, gender, pairs))

                        # Save pairs to CSV and handle knockout tournament
                        folder_name = f"APR_RANDOM_DRAWS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        os.makedirs(folder_name, exist_ok=True)
                        save_pairs_to_csv(folder_name, sport, gender, age_range, pairs)
                        handle_knockout_tournament(pairs, folder_name, sport, gender, participants_df, age_range)

                    elif action == '3':
                        # List existing draws
                        list_draws(draws)

                    elif action == '4':
                        # Add a participant
                        participants_df = add_participant(participants_df, sport)

                    elif action == '5':
                        # Delete a participant
                        participants_df = delete_participant(participants_df, sport, gender)

                    elif action == '6':
                        # Back to main menu
                        break

                    else:
                        print("Invalid option, please try again.")

    except KeyError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

