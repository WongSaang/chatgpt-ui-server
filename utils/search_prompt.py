import re
from typing import List
from datetime import datetime
from .search_abc import SearchResult

def remove_commands(query: str) -> str:
    query = re.sub(r'\/page:(\S+)\s+', '', query)
    query = re.sub(r'\/site:(\S+)\s+', '', query)
    return query


def compile_prompt(results: List[SearchResult], query: str, default_prompt: str) -> str:
    formatted_results = format_web_results(results)
    current_date = datetime.now().strftime("%m/%d/%Y")
    print(default_prompt)
    prompt = replace_variables(default_prompt, {
        '[web_results]': formatted_results,
        '[query]': remove_commands(query),
        '[current_date]': current_date
    })
    return prompt


def format_web_results(results: List[SearchResult]) -> str:
    if len(results) == 0:
        return "No results found.\n"
    formatted_results = ""
    counter = 1
    for result in results:
        formatted_results += f"[{counter}] \"{result.body}\"\nURL: {result.url}\n\n"
        counter += 1
    return formatted_results


def replace_variables(prompt: str, variables: dict) -> str:
    new_prompt = prompt
    for key, value in variables.items():
        try:
            new_prompt = new_prompt.replace(key, value)
        except Exception as error:
            print("Search prompt error: ", error)
    return new_prompt
