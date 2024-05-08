from typing import Optional


class TextParser:
    def __init__(self, lines, length_indentation=5, current_indentation=0):
        self.lines = lines
        self.length_indentation = length_indentation
        self.current_indentation = current_indentation
        self.parsed_text = {}
        self.parsers = {}

    def parse(self):
        subparser_key = None

        lines_for_subparser: Optional[list] = None

        for i, line in enumerate(self.lines):
            if line == '\n':
                continue

            # count number of leading spaces
            spaces = len(line) - len(line.lstrip())

            if lines_for_subparser is not None:
                if spaces > self.current_indentation:
                    lines_for_subparser.append(line)
                    continue
                else:
                    self.parsers[subparser_key] = (
                        TextParser(lines_for_subparser,
                                   length_indentation=self.length_indentation,
                                   current_indentation=self.current_indentation + self.length_indentation))
                    self.parsers[subparser_key].parse()
                    lines_for_subparser = None
                    self.parsed_text[subparser_key] = self.parsers[subparser_key].parsed_text
                    subparser_key = None

            line_no_whitespace = line.replace(' ', '')
            line_no_whitespace = line_no_whitespace.replace('\n', '')
            key, value = line_no_whitespace.split(':')

            if value == '':
                lines_for_subparser = []
                subparser_key = key
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
                self.parsed_text[key] = value

        if lines_for_subparser is not None:
            self.parsers[subparser_key] = (
                TextParser(lines_for_subparser,
                           length_indentation=self.length_indentation,
                           current_indentation=self.current_indentation + self.length_indentation))
            self.parsers[subparser_key].parse()
            self.parsed_text[subparser_key] = self.parsers[subparser_key].parsed_text
