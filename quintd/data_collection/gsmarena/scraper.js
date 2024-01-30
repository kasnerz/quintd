const gsmarena = require('gsmarena-api');
const fs = require('fs');
const path = require('path');

const fetchWithRetry = async (requestFunction, args, delay = 1000000) => {
    let retries = 0;
    let currentDelay = delay;

    while (true) {
        try {
            return await requestFunction(...args);
        } catch (error) {
            if (error.response && error.response.status === 429) {
                console.log(`Too Many Requests. Retrying in ${currentDelay / 1000} seconds...`);
                await new Promise(resolve => setTimeout(resolve, currentDelay));
                currentDelay *= 2;
                retries++;
            } else {
                throw error;
            }
        }
    }
};

const fetchPhone = async (phone) => {
    console.log(`Fetching data for ${phone.name}...`);
    return fetchWithRetry(gsmarena.catalog.getDevice, [phone.id]);
};

const fetchBrand = async (brand, maxPhonesPerBrand) => {
    // Get a list of mobile phones for the current brand
    console.log(`Fetching data for ${brand.name}...`);
    var phones = await fetchWithRetry(gsmarena.catalog.getBrand, [brand.id]);

    var i = 0;
    for (const phone of phones) {
        const details = await fetchPhone(phone);
        phone.details = details;
        console.log(details);

        i++;
        if (i > maxPhonesPerBrand) {
            break;
        }
    }
    // remove all phones without details
    phones = phones.filter(phone => phone.details);

    return phones;
};

// Using an async function to use the 'await' keyword
const fetchData = async (outputPath, maxPhonesPerBrand) => {
    var data = [];
    // Get a list of mobile phone brands
    const brands = await fetchWithRetry(gsmarena.catalog.getBrands, []);

    for (const brand of brands) {
        try {
            var deviceList = await fetchBrand(brand, maxPhonesPerBrand);
            brand.deviceList = deviceList;
            data.push(brand);
        } catch (error) {
            console.error(`Failed to fetch data for ${brand}: ${error.message}`);
        }
    }
    fs.writeFile(outputPath, JSON.stringify(data, null, 4), (err) => {
        if (err) {
            console.error(err);
            return;
        };
        console.log(`Data saved to ${outputPath}`);
    });
};

const outputPath = process.argv[2];
const maxPhonesPerBrand = process.argv[3];
fetchData(outputPath, maxPhonesPerBrand);
