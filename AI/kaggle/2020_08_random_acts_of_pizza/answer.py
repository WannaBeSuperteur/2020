import json

jf = open('test.json', 'r')
json_file = jf.read()
jf.close()
    
json_data = json.loads(json_file)

print(json_data[:3])

result = ''
for i in range(len(json_data)):
    x = json_data[i]["request_id"]
    y = json_data[i]["requester_number_of_posts_on_raop_at_request"]

    print(x, y)
    
    if y <= 1:
        result += str(x) + ',' + str(0) + '\n'
    else:
        result += str(x) + ',' + str(1) + '\n'

f = open('result.csv', 'w')
f.write(result)
f.close()
