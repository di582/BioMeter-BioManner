import docx
from pathlib import Path
import operator
from typing import Annotated, List, Tuple, TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re
import csv



def extract_errors(input_folder, output_folder):
    # Ensure the output directory exists
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create an error summary list
    error_summary = []
    total_errors = 0

    # At the beginning of the extract_errors function, create a processed folder
    processed_folder = output_path / "processed"
    processed_folder.mkdir(parents=True, exist_ok=True)

    # Process all txt files
    for file_path in Path(input_folder).glob("*.txt"):
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Directly create the output file in the processed folder
            output_file = processed_folder / f"{file_path.stem}_errors.txt"

            # More flexible error matching pattern
            error_pattern = re.compile(r'<Error(.*?)Error>', re.DOTALL)
            errors = error_pattern.findall(content)

            # More flexible type extraction pattern
            type_pattern = re.compile(
                r'<Type\s*(.*?)\s*Type>',
                re.DOTALL | re.IGNORECASE
            )

            with open(output_file, 'w', encoding='utf-8') as f_out:
                fact_errors = 0
                rel_errors = 0
                other_errors = 0

                if errors:
                    f_out.write(f"File: {file_path.name}\n")
                    f_out.write(f"Found {len(errors)} errors in total:\n\n")

                    # Process each error block
                    for i, error_content in enumerate(errors, 1):
                        # Extract error type (more robust method)
                        error_type = "Unknown Type"
                        type_matches = type_pattern.findall(error_content)

                        if type_matches:
                            error_type = type_matches[0].strip()

                            # More flexible type matching
                            if "Fact" in error_type or "fact" in error_type.lower():
                                fact_errors += 1
                                error_type = "Factual Error"
                            elif "Relevance" in error_type or "rel" in error_type.lower():
                                rel_errors += 1
                                error_type = "Relevance Error"
                            else:
                                other_errors += 1
                        else:
                            other_errors += 1

                        # Write error details
                        f_out.write(f"Error #{i}:\n")
                        f_out.write(f"Type: {error_type}\n")
                        f_out.write(f"Content: {error_content.strip()}\n\n")

                    # Add statistics
                    f_out.write("\n===== Error Statistics =====\n")
                    f_out.write(f"Factual Errors: {fact_errors}\n")
                    f_out.write(f"Relevance Errors: {rel_errors}\n")
                    f_out.write(f"Other Types of Errors: {other_errors}\n")

                    print(f"√ {file_path.name}: Found {len(errors)} errors")
                    total_errors += len(errors)
                else:
                    # Add detailed debug information
                    f_out.write(f"File: {file_path.name}\n")
                    f_out.write("No error content found\n\n")
                    f_out.write("===== Content Preview =====\n")
                    f_out.write(content[:1000])  # Output the first 1000 characters for debugging
                    print(f"× {file_path.name}: No errors found - added content preview to the output file")

                # Add to error summary
                error_summary.append({
                    'filename': file_path.name,
                    'fact_errors': fact_errors,
                    'rel_errors': rel_errors,
                    'other_errors': other_errors,
                    'total_errors': len(errors)
                })
        except Exception as e:
            print(f"! Processing error: {file_path.name} - {str(e)}")
            error_summary.append({
                'filename': file_path.name,
                'fact_errors': 0,
                'rel_errors': 0,
                'other_errors': 0,
                'total_errors': 0,
                'error': str(e)
            })

    # Generate error summary CSV file
    summary_file = output_path / "errors_summary.csv"
    with open(summary_file, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Filename', 'Number of Factual Errors', 'Number of Relevance Errors', 'Number of Other Errors', 'Total Number of Errors']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in error_summary:
            writer.writerow({
                'Filename': item['filename'],
                'Number of Factual Errors': item['fact_errors'],
                'Number of Relevance Errors': item['rel_errors'],
                'Number of Other Errors': item.get('other_errors', 0),
                'Total Number of Errors': item['total_errors']
            })

    return error_summary

def count_sentences(input_folder, output_folder):
    # Ensure the output directory exists
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a results list
    results = []
    total_sentences = 0

    # Regular expression for error content
    error_pattern = re.compile(r'<Error(.*?)Error>', re.DOTALL)

    # Process all txt files
    for file_path in Path(input_folder).glob("*.txt"):
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Remove the error part (including error tags and content)
            clean_content = error_pattern.sub('', content)

            # Count the number of periods (only counting Chinese periods "。")
            sentence_count = clean_content.count('。')

            # Add to results
            results.append({
                'filename': file_path.name,
                'sentence_count': sentence_count
            })

            total_sentences += sentence_count
            print(f"√ {file_path.name}: Found {sentence_count} periods")

        except Exception as e:
            print(f"! Processing error: {file_path.name} - {str(e)}")
            results.append({
                'filename': file_path.name,
                'sentence_count': 0,
                'error': str(e)
            })

    # Generate CSV report
    csv_path = output_path / "sentence_counts.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Filename', 'Number of Periods']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            writer.writerow({
                'Filename': item['filename'],
                'Number of Periods': item['sentence_count']
            })

    return results, total_sentences

def calculate_correctness_percentage(error_summary_csv, sentence_count_csv, output_folder):
    # Ensure the output directory exists
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read error summary data
    error_data = {}
    with open(error_summary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            error_data[row['Filename']] = {
                'fact_errors': int(row['Number of Factual Errors']),
                'rel_errors': int(row['Number of Relevance Errors']),
                'total_errors': int(row['Total Number of Errors'])
            }

    # Read sentence statistics data
    sentence_data = {}
    with open(sentence_count_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence_data[row['Filename']] = int(row['Number of Periods'])

    # Create a results list
    results = []
    total_sentences = 0
    total_fact_correct = 0
    total_rel_correct = 0
    total_fact_errors = 0
    total_rel_errors = 0

    # For calculating average percentage
    sum_fact_percentage = 0.0
    sum_rel_percentage = 0.0
    sum_composite_percentage = 0.0  # New: sum of composite percentages
    file_count = 0

    # Process all files
    for filename in error_data:
        if filename in sentence_data:
            sentence_count = sentence_data[filename]
            fact_errors = error_data[filename]['fact_errors']
            rel_errors = error_data[filename]['rel_errors']

            # Calculate the number of correct sentences
            fact_correct = sentence_count - fact_errors
            rel_correct = sentence_count - rel_errors

            # Calculate percentage (avoid division by zero)
            fact_percentage = (fact_correct / sentence_count * 100) if sentence_count > 0 else 0
            rel_percentage = (rel_correct / sentence_count * 100) if sentence_count > 0 else 0
            composite_percentage = (fact_percentage + rel_percentage) / 2  # New: composite correctness percentage

            results.append({
                'filename': filename,
                'sentence_count': sentence_count,
                'fact_errors': fact_errors,
                'rel_errors': rel_errors,
                'fact_correct': fact_correct,
                'rel_correct': rel_correct,
                'fact_percentage': fact_percentage,
                'rel_percentage': rel_percentage,
                'composite_percentage': composite_percentage  # New
            })

            # Accumulate totals
            total_sentences += sentence_count
            total_fact_correct += fact_correct
            total_rel_correct += rel_correct
            total_fact_errors += fact_errors
            total_rel_errors += rel_errors

            # Accumulate percentages
            sum_fact_percentage += fact_percentage
            sum_rel_percentage += rel_percentage
            sum_composite_percentage += composite_percentage  # New
            file_count += 1
        else:
            print(f"Warning: File {filename} does not exist in the sentence statistics file")

    # Calculate overall percentage
    overall_fact_percentage = (total_fact_correct / total_sentences * 100) if total_sentences > 0 else 0
    overall_rel_percentage = (total_rel_correct / total_sentences * 100) if total_sentences > 0 else 0
    overall_composite_percentage = (overall_fact_percentage + overall_rel_percentage) / 2  # New

    # Calculate average percentage
    avg_fact_percentage = sum_fact_percentage / file_count if file_count > 0 else 0
    avg_rel_percentage = sum_rel_percentage / file_count if file_count > 0 else 0
    avg_composite_percentage = sum_composite_percentage / file_count if file_count > 0 else 0  # New

    # Generate CSV report - add total row and average row
    csv_path = output_path / "correctness_percentage.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = [
            'Filename',
            'Total Sentences',
            'Number of Factual Errors',
            'Number of Relevance Errors',
            'Number of Factually Correct Sentences',
            'Number of Relevantly Correct Sentences',
            'Factual Correctness Percentage',
            'Relevance Correctness Percentage',
            'Composite Correctness Percentage'  # New
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write data for each file
        for item in results:
            writer.writerow({
                'Filename': item['filename'],
                'Total Sentences': item['sentence_count'],
                'Number of Factual Errors': item['fact_errors'],
                'Number of Relevance Errors': item['rel_errors'],
                'Number of Factually Correct Sentences': item['fact_correct'],
                'Number of Relevantly Correct Sentences': item['rel_correct'],
                'Factual Correctness Percentage': f"{item['fact_percentage']:.2f}%",
                'Relevance Correctness Percentage': f"{item['rel_percentage']:.2f}%",
                'Composite Correctness Percentage': f"{item['composite_percentage']:.2f}%"  # New
            })

        # Add overall statistics row
        writer.writerow({
            'Filename': "Overall Statistics",
            'Total Sentences': total_sentences,
            'Number of Factual Errors': total_fact_errors,
            'Number of Relevance Errors': total_rel_errors,
            'Number of Factually Correct Sentences': total_fact_correct,
            'Number of Relevantly Correct Sentences': total_rel_correct,
            'Factual Correctness Percentage': f"{overall_fact_percentage:.2f}%",
            'Relevance Correctness Percentage': f"{overall_rel_percentage:.2f}%",
            'Composite Correctness Percentage': f"{overall_composite_percentage:.2f}%"  # New
        })

        # Add average statistics row
        writer.writerow({
            'Filename': "Average Statistics",
            'Total Sentences': "",
            'Number of Factual Errors': "",
            'Number of Relevance Errors': "",
            'Number of Factually Correct Sentences': "",
            'Number of Relevantly Correct Sentences': "",
            'Factual Correctness Percentage': f"{avg_fact_percentage:.2f}%",
            'Relevance Correctness Percentage': f"{avg_rel_percentage:.2f}%",
            'Composite Correctness Percentage': f"{avg_composite_percentage:.2f}%"  # New
        })

    # Generate overall statistics report
    summary_path = output_path / "overall_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("===== Overall Statistics Results =====\n")
        f.write(f"Total number of files processed: {len(results)}\n")
        f.write(f"Total number of sentences: {total_sentences}\n")
        f.write(f"Total factual errors: {total_fact_errors}\n")
        f.write(f"Total relevance errors: {total_rel_errors}\n")
        f.write(f"Total factually correct: {total_fact_correct}\n")
        f.write(f"Total relevantly correct: {total_rel_correct}\n")
        f.write(f"Overall factual correctness percentage: {overall_fact_percentage:.2f}%\n")
        f.write(f"Overall relevance correctness percentage: {overall_rel_percentage:.2f}%\n")
        f.write(f"Overall composite correctness percentage: {overall_composite_percentage:.2f}%\n")  # New
        f.write("\n===== Average Statistics Results =====\n")
        f.write(f"Average factual correctness percentage: {avg_fact_percentage:.2f}%\n")
        f.write(f"Average relevance correctness percentage: {avg_rel_percentage:.2f}%\n")
        f.write(f"Average composite correctness percentage: {avg_composite_percentage:.2f}%\n")  # New
        f.write(f"Calculation method: Sum of all file percentages divided by the number of files\n")

    return results, {
        'total_sentences': total_sentences,
        'total_fact_correct': total_fact_correct,
        'total_rel_correct': total_rel_correct,
        'overall_fact_percentage': overall_fact_percentage,
        'overall_rel_percentage': overall_rel_percentage,
        'overall_composite_percentage': overall_composite_percentage,  # New
        'avg_fact_percentage': avg_fact_percentage,
        'avg_rel_percentage': avg_rel_percentage,
        'avg_composite_percentage': avg_composite_percentage  # New
    }

def calculate_error_ratios(input_csv, output_folder):
    """
    Calculate the ratio of factual and relevance errors in error files

    Parameters:
    input_csv - Path to the CSV file containing file labels
    output_folder - Output directory
    """
    # Ensure the output directory exists
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read label data
    error_files = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only process rows with label 0 (files with errors)
            if row['Label'] == '0':
                # Get the number of errors
                fact_errors = int(row['Number of Factual Errors'])
                rel_errors = int(row['Number of Relevance Errors'])
                total_errors = int(row['Total Number of Errors'])

                # Calculate ratio (avoid division by zero)
                fact_ratio = (fact_errors / total_errors * 100) if total_errors > 0 else 0
                rel_ratio = (rel_errors / total_errors * 100) if total_errors > 0 else 0

                # Add to results list
                error_files.append({
                    'filename': row['Filename'],
                    'fact_errors': fact_errors,
                    'rel_errors': rel_errors,
                    'total_errors': total_errors,
                    'fact_ratio': fact_ratio,
                    'rel_ratio': rel_ratio
                })

    # Generate results file
    result_csv = output_path / "error_ratios.csv"
    with open(result_csv, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Filename', 'Number of Factual Errors', 'Number of Relevance Errors', 'Total Number of Errors',
                      'Factual Error Ratio (%)', 'Relevance Error Ratio (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in error_files:
            writer.writerow({
                'Filename': item['filename'],
                'Number of Factual Errors': item['fact_errors'],
                'Number of Relevance Errors': item['rel_errors'],
                'Total Number of Errors': item['total_errors'],
                'Factual Error Ratio (%)': f"{item['fact_ratio']:.2f}",
                'Relevance Error Ratio (%)': f"{item['rel_ratio']:.2f}"
            })

    # Calculate average ratio
    total_fact_ratio = sum(item['fact_ratio'] for item in error_files)
    total_rel_ratio = sum(item['rel_ratio'] for item in error_files)
    avg_fact_ratio = total_fact_ratio / len(error_files) if error_files else 0
    avg_rel_ratio = total_rel_ratio / len(error_files) if error_files else 0

    # Generate statistics report
    report_path = output_path / "error_ratios_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("===== Error Ratio Analysis Report =====\n")
        f.write(f"Number of files analyzed: {len(error_files)}\n")
        f.write(f"Average factual error ratio: {avg_fact_ratio:.2f}%\n")
        f.write(f"Average relevance error ratio: {avg_rel_ratio:.2f}%\n")
        f.write("\nCalculation Method Explanation:\n")
        f.write("  Factual Error Ratio = (Number of Factual Errors / Total Number of Errors) × 100%\n")
        f.write("  Relevance Error Ratio = (Number of Relevance Errors / Total Number of Errors) × 100%\n")
        f.write("  Note: If the total number of errors is 0, the ratio is set to 0%\n")

    return error_files, avg_fact_ratio, avg_rel_ratio


def calculate_perfect_files_percentage(error_summary_csv, output_folder):
    """
    Calculate the percentage of perfect files (without any errors)

    Parameters:
    error_summary_csv - Path to the error summary CSV file
    output_folder - Output directory
    """
    # Ensure the output directory exists
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read error summary data
    perfect_files = 0
    total_files = 0
    file_labels = []

    with open(error_summary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_files += 1

            # Get the number of errors
            fact_errors = int(row['Number of Factual Errors'])
            rel_errors = int(row['Number of Relevance Errors'])
            total_errors = int(row['Total Number of Errors'])

            # Determine the label: 1 for no errors, 0 for errors
            if fact_errors == 0 and rel_errors == 0 and total_errors == 0:
                label = 1
                perfect_files += 1
            else:
                label = 0

            file_labels.append({
                'filename': row['Filename'],
                'fact_errors': fact_errors,
                'rel_errors': rel_errors,
                'total_errors': total_errors,
                'label': label
            })

    # Calculate the percentage of perfect files
    perfect_percentage = (perfect_files / total_files * 100) if total_files > 0 else 0

    # Generate results file
    result_csv = output_path / "file_labels.csv"
    with open(result_csv, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Filename', 'Number of Factual Errors', 'Number of Relevance Errors', 'Total Number of Errors', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in file_labels:
            writer.writerow({
                'Filename': item['filename'],
                'Number of Factual Errors': item['fact_errors'],
                'Number of Relevance Errors': item['rel_errors'],
                'Total Number of Errors': item['total_errors'],
                'Label': item['label']
            })

    # Generate statistics report
    report_path = output_path / "perfect_files_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("===== Perfect File Analysis Report =====\n")
        f.write(f"Total number of files: {total_files}\n")
        f.write(f"Number of perfect files (no errors): {perfect_files}\n")
        f.write(f"Percentage of perfect files: {perfect_percentage:.2f}%\n")
        f.write("\nLabel Explanation:\n")
        f.write("  0: File contains at least one factual or relevance error\n")
        f.write("  1: File has no errors (perfect file)\n")

    return perfect_percentage, file_labels



class Execute(TypedDict):
    """
    Type definition for storing execution state
    past: Stores the execution history, containing the results of all processing steps
    """
    past: Annotated[List[Tuple], operator.add]  # Store execution history

# 1. Initialize the model and tools, define and bind the tools to the model, let the large model decide whether to call the tools
model = ChatOpenAI(
    model="qwen-max",  # Use the Qwen Plus model
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",  # DashScope's API address
    openai_api_key="your API key",  # Replace with your API key
    extra_body={
        "enable_search": True  # Enable web search functionality
    }
)

model2 = ChatOpenAI(
    model="qwen-turbo",  # Use the Qwen Plus model
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",  # DashScope's API address
    openai_api_key="your API key",  # Replace with your API key
)

# Define the first prompt template - for text chunking and initial annotation
prompt1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a professional text annotation assistant. Your task is to annotate the input question and answer text according to the following format.

            Annotation Rules:
            1. Add <Totalscore Overall Relevance Score:XX, Overall Correctness Score:XX Totalscore> after the question.
            2. Segment the answer according to the following rules:
               - If there is a clear content logic, segment by content logic.
               - If there is no clear content logic but there are obvious headings (e.g., "I.", "1."), segment by headings.
               - If a segment is too short, i.e., less than 50 words, merge multiple segments.
            3. Enclose each segment with tags like <par1>, <par2>, <par3>, etc.
            4. In each <par>, add <Subscore Relevance Score:XX, Correctness Score:XX Subscore> and a summary of the segment, for example: <par1 Summary of the first part's content <Subscore Relevance Score:XX, Correctness Score:XX Subscore>
            5. If the question and answer text is in English, do not translate it into Chinese.
            
            Example Input:
            Question content
            Answer part one
            Answer part two

            Example Output:
            <Question1 Question content<Totalscore Overall Relevance Score:XX, Overall Correctness Score:XX Totalscore>
            <par1 First part title<Subscore Relevance Score:XX, Correctness Score:XX Subscore>
            Answer part one content
            par1>
            <par2 Second part title<Subscore Relevance Score:XX, Correctness Score:XX Subscore>
            Answer part two content
            par2>
            Question1>

            Note:
            1. Do not change the original content, including the Question tag.
            2. Keep XX unchanged, do not replace it with specific values.
            3. Ensure all tags are closed, in the format <par  par>.
            4. Add ANSWER1 at the end of the response.

            Now, please process the following question and answer text:"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the second prompt template - for error checking and scoring
prompt2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are an expert in the field of fermentation. Your task is to check for errors in the annotated text and score it.

            Scoring and Error Annotation Rules:
            1. Check the content of each paragraph (par):
               - Relevance: Whether the content is relevant to the question.
               - Correctness: Whether the content has factual errors.
            
            2. When an error is found, add an error annotation at the corresponding position in the original text:
               <Error Error content <Type Error type, Specific reason for the error Type>Error>
               - The specific reason for the error is a detailed description of the error.
               - The error type can be "Factual Error" or "Relevance Error".
               - The annotation must be between the <par and par> tags.
            
            3. Scoring Rules:
               - If a paragraph has no errors, give both relevance and correctness a score of 100.
               - If an error is found, reduce the score accordingly based on the severity of the error.
               - Fill the score into the XX position of the Subscore tag in the corresponding paragraph.
               - Keep the XX in Totalscore unchanged.
            
            4. Formatting Requirements:
               - Keep the original paragraph division unchanged.
               - Ensure all tags are correctly closed.
               - Retain the original error annotations.
               - There should be a blank line between paragraphs.
               - Do not have ANSWER1 in the output.
            
            5. At the end of the text, state the total number of errors found.
            
            Now, please process the following annotated text:'''
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the first agent function - for text chunking and initial annotation
def agent1(state: Execute) -> dict:
    """
    First stage processing function: chunks and initially annotates the text
    Parameters:
        state: The current execution state, containing the input message
    Returns:
        A dictionary containing the processing result
    """
    try:
        # Get the input message
        messages = state.get("past", [])
        if not messages:
            raise ValueError("No input message received")
            
        # Get the content of the last message
        last_message = messages[-1]
        if isinstance(last_message, str):
            content = last_message
        else:
            content = last_message.content
            
        print(f"\nProcessing text content:\n{content}\n")
        
        # Create the processing chain (prompt template + model)
        processing_chain = prompt1 | model2
        


        raw = processing_chain.invoke({"messages": [HumanMessage(content=content)]})
        answer_text = raw if isinstance(raw, str) else raw.content
        if "ANSWER1" not in answer_text:
            answer_text += " ANSWER1"
        print(f"\nFirst step processing result:\n{answer_text}")
        return {"past": [answer_text]}

    except Exception as e:
        print(f"agent1 processing error: {str(e)}")
        raise

# Define the second agent function - for error checking and scoring
def agent2(state: Execute) -> dict:
    """
    Second stage processing function: performs error checking and scoring
    
    Parameters:
        state: The current execution state, containing the result of the first stage
        
    Returns:
        A dictionary containing the processing result
    """
    try:
        # Get the result of the first stage
        messages = state.get("past", [])
        if not messages:
            raise ValueError("Did not receive the result of the first stage")
            
        # Get the content of the last message
        last_message = messages[-1]
        if isinstance(last_message, str):
            content = last_message
        else:
            content = last_message.content
            
        # Create the processing chain (prompt template + model)
        processing_chain = prompt2 | model
        
        # Call the model for error checking and scoring
        response = processing_chain.invoke({
            "messages": [HumanMessage(content=content)]
        })
        
        print(f"\nSecond step processing result:\n{response.content}")
        
        # Return the processing result
        return {
            "past": [response]
        }
    except Exception as e:
        print(f"agent2 processing error: {str(e)}")
        raise

# Define the state judgment function
def should_continue(state: Execute) -> str:
    """
    Function to decide the next step of the workflow
    
    Parameters:
        state: The current execution state
        
    Returns:
        The name of the next operation: 'chunk', 'annotate' or 'end'
    """
    try:
        messages = state.get("past", [])  # Get historical messages
        if not messages:  # If there are no messages, start with chunking
            print("\nStarting the first step...")
            return "chunk"
        
        last_message = messages[-1]  # Get the last message
        if isinstance(last_message, str):  # Check if last_message is a string type
            last_content = last_message    # If it is a string, use it directly
        else:
            last_content = last_message.content  # If not a string, get its content attribute

        # Decide the next operation based on the message content
        if "ANSWER1" in last_content:
            print("\nFirst step completed...")
            return "annotate"  # Perform error annotation
        else:
            return "end"  # End processing
        
    except Exception as e:
        print(f"State judgment error: {str(e)}")
        raise

# 2. Initialize the graph with the state, define an empty graph, MessagesState is used to store messages
workflow = StateGraph(Execute)
# 3. Define the graph nodes, define the two nodes we will loop through, add the agent and tools nodes
workflow.add_node("chunk", agent1)
workflow.add_node("annotate", agent2)

# 4. Define the entry point and graph edges
# Set the entry point to "agent"
# This means this is the first node to be called
workflow.set_entry_point("chunk")

# Add conditional edges
workflow.add_conditional_edges(
    "chunk",
    should_continue,
    {
        "chunk": "chunk",
        "annotate": "annotate",
        "end": END
    }
)

workflow.add_edge("annotate", END)

# 5. Compile the graph
# This compiles it into a LangChain runnable object,
# which means you can use it like any other runnable object.
# Note that we (optionally) pass memory when compiling the graph
app = workflow.compile()
def read_docx(file_path: str) -> list:
    """
    Extract question-answer pairs separated by "//" from a Word document and wrap them with <Question i ... Question i> tags
    
    Parameters:
        file_path: Path to the Word document
        
    Returns:
        A list of question-answer pairs, where each element is a question-answer pair string in the format <Question i Question\nAnswer\nQuestion i>
    """
    doc = docx.Document(file_path)
    qa_strings = []
    current_block = []
    qa_index = 1  # Question number, starting from 1
    
    for para in doc.paragraphs:
        text = para.text.strip()
        
        if not text:
            continue
            
        if text == "//":
            if current_block:
                # Extract question and answer
                question = current_block[0].strip()
                answer = "\n".join(current_block[1:]).strip()
                
                # Ensure neither question nor answer is empty
                if question and answer:
                    # Construct the question-answer pair string, adding <Question i ... Question i> tags
                    qa_str = f"<Question {qa_index} {question}\n{answer}\nQuestion {qa_index}>"
                    qa_strings.append(qa_str)
                    qa_index += 1
                else:
                    print(f"Warning: Skipping empty question-answer pair")
                current_block = []
        else:
            current_block.append(text)
    
    # Process the last question-answer block
    if current_block:
        question = current_block[0].strip()
        answer = "\n".join(current_block[1:]).strip()
        if question and answer:
            qa_str = f"<Question {qa_index} {question}\n{answer}\nQuestion {qa_index}>"
            qa_strings.append(qa_str)
    
    # Print the number of question-answer pairs read and an example
    print(f"Read {len(qa_strings)} question-answer pairs")
    if qa_strings:
        print(f"Example question-answer pair:\n{qa_strings[0][:200]}...")
    
    return qa_strings



def main() -> None:
    try:
        docs_dir = Path("docs")
        output_dir = Path("output")
        docs_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        docx_files = sorted(docs_dir.glob("*.docx"))  # Keep the order
        if not docx_files:
            print("No docx files found")
            return
    
        for idx, doc_file in enumerate(docx_files, start=1):
            # 1. Generate a dedicated result directory for the current docx
            result_subdir = output_dir / f"result_dir{idx}"
            result_subdir.mkdir(parents=True, exist_ok=True)

            # 2. Read question-answer pairs
            qa_pairs = read_docx(doc_file)
            if not qa_pairs:
                print(f"{doc_file.name} has no valid question-answers, skipping")
                continue

            # 3. Randomly sample and process
            #selected = random.sample(qa_pairs, min(1, len(qa_pairs)))
            for i, pair in enumerate(qa_pairs, 1):
                try:
                    qa_file = result_subdir /'qa'
                    qa_file.mkdir(parents=True, exist_ok=True)
                    res = app.invoke({"past": [HumanMessage(content=pair)]})
                    txt_file = qa_file/f"qa_{i:03d}.txt"
                    txt_file.write_text(res["past"][-1].content, encoding="utf-8")
                except Exception as e:
                    print(f"Error processing pair {i} of {doc_file.name}: {e}")

            # 4. Immediately perform statistics within the current result_dirX
            try:
                extract_errors(qa_file, result_subdir)
                count_sentences(qa_file, result_subdir)
                calculate_correctness_percentage(
                    result_subdir / "errors_summary.csv",
                    result_subdir / "sentence_counts.csv",
                    result_subdir
                )
                calculate_perfect_files_percentage(
                    result_subdir / "errors_summary.csv",
                    result_subdir
                )
                calculate_error_ratios(
                    result_subdir / "file_labels.csv",
                    result_subdir
                )
                
                print(f"[Completed] {doc_file.name} -> {result_subdir.name}")
            except Exception as e:
                print(f"[Statistics Failed] {doc_file.name}: {e}")

    except Exception as e:
        print(f"Main process exception: {e}")

if __name__ == "__main__":
    main()
