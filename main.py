"""
Calculating all possible solutions to the 24 game and writing them to a file.

R: range of the numbers
O: the number of operations
C: the number of cards

O( (R**C) * (O**(C-1)) * ((C-1)**(C-1)) )
"""

from datetime import datetime
from time import perf_counter
from collections import defaultdict
from sympy import together
from typing import Any
from collections.abc import Generator
import os
import csv


def verified_input(prompt: str, target_type: type, *criterias: str) -> Any:
    """
    Prompts the user and verifies that the response can be turned into the target type and passes the criteria.
    If it does not valid, the user will be prompted again with the prompt

    Note that there can be as many criteria as is needed and that it needs to be a conditional statement in the form of
    a string that will be evaluated.
    """
    while True:
        the_input = input(prompt)
        try:
            the_input = target_type(the_input)
            for criteria in criterias:
                assert eval(criteria)
            return the_input
        except Exception as e:
            print(e)
            print(
                f'Sorry, "{the_input}" is an invalid input. Please try again.')


# Approximately after how long would you like to be informed of progress
TARGET = verified_input(
    'What should the target value be in this calculation?\n>>> ', int)
MIN_NUM = verified_input(
    'What should the minimum value be (this is inclusive)?\n>>> ', int)
MAX_NUM = verified_input(
    'What should the maximum value be (this is inclusive)?\n>>> ', int, f'the_input > {MIN_NUM}')
POSSIBLE_NUMS = [str(i) for i in range(MIN_NUM, MAX_NUM + 1)]
OPS = ['+', '-', '*', '/']  # % and ** could potentially be added
NUM_CARDS = verified_input('How many values need to be used to find the target value?\n>>> ', int,
                           f'the_input > 2')
# a rough estimate of how long it will take to calculate an equation in seconds
TIME_PER_EQ = 2e-5

MAX_EQ_VAL = int(((NUM_CARDS - 2) ** 2 * ((NUM_CARDS - 1) ** (NUM_CARDS - 1) - 1) -
                  (NUM_CARDS - 1) * (NUM_CARDS ** 2 * (NUM_CARDS - 1) ** (NUM_CARDS - 2) -
                                     4 * NUM_CARDS * (NUM_CARDS - 1) ** (NUM_CARDS - 2) + 3 * (NUM_CARDS - 1) ** (
                      NUM_CARDS - 2) + 1)
                  ) / (NUM_CARDS - 2) ** 2)  # ~ O((c-1)**(c-1))
# the above is equivalent to:
#     sum(
#         (NUM_CARDS - 2 - i) * (NUM_CARDS - 1) ** i
#         for i in range(NUM_CARDS - 2)
#     )

rough_estimate_of_eqs = \
    (MAX_NUM - MIN_NUM + 1) ** NUM_CARDS * \
    len(OPS) ** (NUM_CARDS - 1) * MAX_EQ_VAL
print(f'There will be approximately be ' +
      f'{rough_estimate_of_eqs:,} ' +
      f'equations to calculate, which means it should take around {rough_estimate_of_eqs * TIME_PER_EQ:,.3f}')
PROGRESS_INTERVAL = .05


def main():
    metadata: list[list[str], list] = [
        ['MIN', 'MAX', 'CARDS', 'OPERATIONS', 'TARGET'],
        [MIN_NUM, MAX_NUM, NUM_CARDS, str(OPS).replace(" ", "")[1:-1], TARGET]
    ]

    solutions: defaultdict[tuple[int], set[str]] = defaultdict(set)

    # generating templates:
    start = perf_counter()
    templates = generate_templates()
    eqs = (MAX_NUM - MIN_NUM + 1) ** NUM_CARDS * len(templates)
    time_taken = perf_counter() - start
    print(f'Time taken to generate templates: {time_taken:.3f} secs; ' +
          f'number of templates generated: {len(templates):,}; ' +
          f'average time per template: {time_taken / len(templates):.3f} secs; ' +
          f'equations to compute: {eqs:,} ' +
          f'estimated time: {eqs * TIME_PER_EQ:,.3f}')

    # generating equations:
    start = perf_counter()
    eq_generator = generate_eqs(templates)
    increment = 100 / eqs
    progress = 0
    interval = PROGRESS_INTERVAL
    solutions_num = 0
    for eq, nums in eq_generator:
        progress += increment
        if progress > interval:
            interval += PROGRESS_INTERVAL
            print(
                f'Progress: {progress:.3f}%; eta: {(100 - progress) * (perf_counter() - start) / progress:,.3f} secs')
        try:
            val = eval(eq)
        except (ZeroDivisionError, TypeError):
            continue
        if val == TARGET and val:
            solutions[nums].add(eq)
            solutions_num += 1
    time_taken = perf_counter() - start
    print(
        f"Time: {time_taken:,.3f} secs; " +
        f"average time per equation: {time_taken / eqs} secs"
    )

    # remove repeated operations like: 8*(9 - 7) + 8; 8 + 8*(9 - 7)
    start = perf_counter()
    for kvp in solutions.items():
        nums, eqs = kvp
        replacement_vals: dict[str: str] = {
            str(nums[i]): chr(int(nums[i]) - MIN_NUM + 97) for i in range(len(nums) - 1, -1, -1)
        }

        unique_expressions = set()
        simplified_expressions = set()
        for expression in eqs:
            tmp = expression[:]
            for replacement_val in replacement_vals.items():
                tmp = tmp.replace(*replacement_val)
            simplified = together(tmp)
            if simplified not in simplified_expressions:
                simplified_expressions.add(simplified)
                unique_expressions.add(expression)

        solutions[nums] = unique_expressions

    solutions_list = sorted(solutions.items(), key=lambda x: x[0])
    solutions_list = [[nums, *eqs] for nums, eqs in solutions_list]

    print(f'Time removing duplicates: {perf_counter() - start:,.3f}')

    # writing data into a csv file
    write_to_csv(
        metadata + solutions_list,
        f'solutions/{TARGET}_{datetime.now().strftime("%D_%H-%M-%S").replace("/", "_")}.csv'
    )


def write_to_csv(contents: list, file_path: str):
    """Writes the given contents into the file at the given file path as a csv file."""
    directory = os.path.dirname(file_path)
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(contents)


def generate_templates():
    generated_poses = set()
    max_vals = list(range(NUM_CARDS - 2, -1, -1))
    for eq_n in range(MAX_EQ_VAL + 1):
        vals = tuple(
            sorted(conv_bases(eq_n, [i for i in range(
                NUM_CARDS)], NUM_CARDS - 1, NUM_CARDS - 1), reverse=True)
        )
        if vals in generated_poses:
            continue
        for nums in zip(vals, max_vals):
            if nums[0] > nums[1]:
                break
        else:
            generated_poses.add(vals)

    ops = set()
    # O(O**(C-1))
    for op_n in range(len(OPS) ** (NUM_CARDS - 1)):
        op = tuple(conv_bases(op_n, OPS, len(OPS), NUM_CARDS - 1))
        ops.add(op)

    templates = set()
    empty_template = [chr(i + 97) for i in range(NUM_CARDS)]
    for pos in generated_poses:
        for op in ops:
            template = empty_template[:]
            for pos_op in zip(pos, op):
                template.insert(*pos_op)
            eq = ''.join(prefix_to_infix(template))
            simplified_eq = str(together(eq))
            if simplified_eq in templates:
                continue
            if not all((simplified_eq.find(char) != -1 and simplified_eq.count(char) == 1) for char in empty_template):
                if eq not in templates:
                    templates.add(eq)
                continue
            templates.add(simplified_eq)
    return templates


def generate_eqs(templates: set[str]) -> \
        Generator[float, None, None] | Generator[tuple[str, tuple[int], float], None, None]:
    """A generator that generates all unique equations by filling in numbers into the template."""

    # O(R**C)
    for num_n in range((MAX_NUM - MIN_NUM + 1) ** NUM_CARDS):
        nums = conv_bases(num_n, POSSIBLE_NUMS, MAX_NUM -
                          MIN_NUM + 1, NUM_CARDS)

        for expression in templates:
            for i, num in zip(range(NUM_CARDS), nums):
                expression = expression.replace(chr(i + 97), num)

            yield expression, tuple(sorted(int(n) for n in nums))


def conv_bases(n: int, conv_key: list | tuple | str | dict, base_of_num: int, target_digits: int) -> list[str]:
    """Converts a number from base ten to operations based off of a conversion list."""
    out = []
    while n != 0:
        n, r = divmod(n, base_of_num)
        out.append(conv_key[r])
    if len(out) <= target_digits - 1:
        out.extend([conv_key[0]] * (target_digits - len(out)))
    return out[::-1]  # list was originally reversed


def prefix_to_infix(equation: list[str]) -> list[str]:
    """Changes an equation from prefix notation to infix notation."""

    def is_operator(char):
        return char in OPS

    stack: list[list[str]] = []

    # read prefix in reverse order
    i = len(equation) - 1
    while i >= 0:
        if not is_operator(equation[i]):  # symbol is operand
            stack.append([equation[i]])
        else:  # symbol is operator
            try:
                stack.append(
                    ["(", *stack.pop(), equation[i], *stack.pop(), ")"])
            except IndexError:
                print(equation, i)
        i -= 1

    return stack[0]


if __name__ == '__main__':
    start_time = perf_counter()
    main()
    print(f'Total time taken: {perf_counter() - start_time}s')
