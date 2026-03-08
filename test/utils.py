raw = [" Bob ", "sue", "SUE", "tim"]
# Bob, sue, tim,  pam, Sue
# [" Bob ", "sue", "SUE", "tim"]
def clean_names(raw_str: str | list) -> list:
    if isinstance(raw_str, str):
        items = raw_str.split(',')
    elif isinstance(raw_str, list):
        items = raw_str
    else:
        pass

    result = []
    for element in items:
        cleaned_element = element.strip().lower()

        if cleaned_element and cleaned_element not in result:
            result.append(element)

    return result


def filter_students(names: list, stop_name: str = 'tim', exclude_list: list = None):
    '''
    Docstring for filter_students
    
    names = ['bob', 'sue', 'tim', 'pam']
    return -> ['bob', 'sue']
    '''
    result = []
    
    if exclude_list is not None:
        normalized_exclude = [x.strip().lower() for x in exclude_list]
    else:
        normalized_exclude = []

    for name in names:
        norm_name = name.strip().lower()

        if norm_name == stop_name:
            break

        if norm_name in normalized_exclude:
            continue

        if norm_name in [x.strip().lower() for x in result]:
            continue

        result.append(norm_name.title())

    return result


def parse_students(raw_students: list) -> list[dict]:
    result = []
    
    def process_part(parts):
        name = parts[0].strip().title()

        try:
            age = int(parts[1].strip())
        except ValueError:
            age = None

        dept = parts[2].strip() or 'Unknwown'

        return {
            'name': name,
            'age': age,
            'dept': dept
        }
    
    for element in raw_students:
        parts = element.split(',')

        if not parts[0].strip():
            continue

        result.append(process_part(parts = parts))

    return result
# obj = filter_students(names = ['bob', 'bob', 'sue', 'tim', 'alex'],
#     stop_name='alex',
#     exclude_list=['sue']
#     )
# obj2 = filter_students(clean_names(raw_str=raw))
o3 = parse_students(raw_students = [
    "Bob, 20, CS",
    "Sue,19,Math",
    " Tim , 21 , Physics",
    "Bob,20,CS",
    "Alex, , Biology",
    "Pam, 22, "
])
print(o3)




'''
clean_names("Bob, sue, tim,  pam, Sue")
# ['bob', 'sue', 'tim', 'pam']

clean_names([" Bob ", "sue", "SUE", "tim"])
# ['bob', 'sue', 'tim']

'''
# Βρογχοι -> Συνεχη εκτελεση μιας διαδικασιας μεχρι η συνθηκη να γινει αληθης


    # if stop_name not in names:
    #     return 'Stop user not in names list'

    # for name in names:
    #     if name == stop_name:
    #         break
    #     elif exclude_list is not None and name.title() in exclude_list:
    #         continue
    #     elif exclude_list is None and name.title() in result:
    #         continue
    #     result.append(name.title())

    # if exclude_list is not None:
    #     for exclude_name in exclude_list:
    #         if exclude_name.title() in result:
    #             result.pop(result.index(exclude_name.title()))
    # else:
    #     pass
