import requests
from bs4 import BeautifulSoup
from csv import writer
reviews_num = 0

# Begin writing to reviews.csv file
with open('reviews.csv', 'w', encoding='utf8', newline='') as f:
    w = writer(f)
    header = ['comapnyName', 'datePublished', 'ratingValue',
              'reviewBody']
    w.writerow(header)

    # Iterate over the 40 pages of reviews
    for i in range(1, 40):
        trust_pilot_base_url = 'https://www.trustpilot.com'
        path = 'review/udemy.com'
        page_num = '?page={}'.format(i)
        url = trust_pilot_base_url + '/' + path + page_num
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')

        reviews = soup.find_all('section',
                                class_='styles_reviewContentwrapper__zH_9M'
                                )
        # Iterate over each review and collect the datePublished, ratingValue and reviewBody
        for review in reviews:
            try:
                comapnyName = 'Udemy'
                datePublished = review.find('time')['datetime']
                ratingValue = review.find('img')['alt'].split()[1]
                reviewBody = review.find('p',
                                         {'class': 'typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn'
                                          }).text.replace('\n', '')
                info = [comapnyName, datePublished, ratingValue, reviewBody]
                w.writerow(info)
                reviews_num += 1
            # Added try catch for AttributeError in the event that one of the variables
            # above are not defined and thus have no atribute text
            except AttributeError:
                continue
print("Total Reviews: " + str(reviews_num))
