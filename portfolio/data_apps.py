import csv
import io
import re
from .validation import ValidationError

def parse(data):
    source = data.get('csv')
    if not isinstance(source, str) or not source.strip() or len(source.encode('utf-8')) > 1000000:
        raise ValidationError('Provide a nonempty UTF-8 CSV of at most 1 MB')
    try:
        reader = csv.reader(io.StringIO(source.lstrip('\ufeff')), strict=True)
        headers = [h.strip() for h in next(reader)]
        if not 1 <= len(headers) <= 50 or any((not h for h in headers)) or len(set(headers)) != len(headers):
            raise ValidationError('Use 1–50 unique, nonempty column names')
        records = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(headers):
                raise ValidationError(f'CSV line {reader.line_num}: expected {len(headers)} fields, got {len(row)}')
            records.append(row)
            if len(records) > 5000:
                raise ValidationError('CSV limit is 5,000 data rows')
        if not records:
            raise ValidationError('Include at least one data row')
        return (headers, records)
    except (csv.Error, StopIteration) as error:
        raise ValidationError(f'Invalid CSV: {error}') from error

def safe_csv(headers, records, *, include_header=True):

    def safe(value):
        if re.fullmatch('-[0-9]+(?:\\.[0-9]+)?', value):
            return value
        return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) or value.startswith(('\t', '\r')) else value
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    if include_header:
        writer.writerow([safe(h) for h in headers])
    writer.writerows([[safe(value) for value in row] for row in records])
    return output.getvalue()
