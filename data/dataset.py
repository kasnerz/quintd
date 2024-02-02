#!/usr/bin/env python3

import json
import random
from pathlib import Path
from collections import defaultdict

# from scripts.openweather.main import get_weather_from_json
import pandas as pd
import json2table
import dateutil.parser
from datetime import datetime
from tinyhtml import h
import string


class Dataset:
    def __init__(self, name, base_path):
        self.base_path = base_path
        self.data_path = f"{self.base_path}/data"
        self.output_path = f"{self.base_path}/outputs"
        self.name = name

        self.outputs = self.load_generated_outputs()

    def load_generated_outputs(self):
        outputs = defaultdict(dict)

        for split in ["dev", "test"]:
            setups = Path.glob(Path(self.output_path) / split / self.name, "*")
            for setup in setups:
                outs = Path.glob(setup, "*.json")
                setup_name = setup.name

                outputs[split][setup_name] = []

                for out in outs:
                    with open(out) as f:
                        outputs[split][setup_name].append(json.load(f))

        return outputs

    def get_data(self, split):
        with open(f"{self.data_path}/{self.name}/{split}.json") as f:
            data = json.load(f)
        return data

    def render(self, table):
        html = json2table.convert(
            table,
            build_direction="LEFT_TO_RIGHT",
            table_attributes={
                "class": "table table-sm caption-top meta-table table-responsive font-mono rounded-3 table-bordered"
            },
        )

        return html

    def get_generated_outputs(self, split, output_idx):
        outs_all = []

        for outs in self.outputs[split].values():
            for model_out in outs:
                out = {}

                out["model"] = model_out["model"]
                out["setup"] = model_out["setup"]
                out["generated"] = None

                if output_idx < len(model_out["generated"]):
                    out["generated"] = model_out["generated"][output_idx]["out"]

                outs_all.append(out)

        return outs_all


class OpenWeather(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, name="openweather")

    def get_data(self, split):
        with open(f"{self.data_path}/{self.name}/{split}.json") as f:
            data = json.load(f)

        forecasts = data["forecasts"]
        data = []

        # https://openweathermap.org/api/hourly-forecast, using metric system
        units = {
            "temp": "°C",
            "wind": "m/s",
            "pressure": "hPa",
            # "humidity": "%",
            "rain": "mm",
            "snow": "mm",
            # "visibility": "m",
        }

        for forecast in forecasts:
            city = forecast["city"]
            lst_filtered = []
            timezone_shift_sec = city["timezone"]

            for key in ["sunrise", "sunset", "population", "timezone"]:
                city.pop(key, None)

            for i, f in enumerate(forecast["list"]):
                # 6-hour intervals
                if i % 2 != 0:
                    continue
                f = {
                    k: v
                    for k, v in f.items()
                    if k not in ["dt", "pop", "sys", "visibility"]
                }

                # remove the main -> temp_kf key
                f["main"] = {
                    k: v
                    for k, v in f["main"].items()
                    if k not in ["temp_kf", "humidity", "sea_level", "grnd_level"]
                }

                # convert "dt_txt" to timestamp
                local_datetime = dateutil.parser.parse(f["dt_txt"])
                local_datetime = local_datetime.timestamp()
                # shift timezone
                local_datetime += timezone_shift_sec
                # convert back to "2023-11-28 09:00:00"
                local_datetime = datetime.fromtimestamp(local_datetime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                f["dt_txt"] = local_datetime

                lst_filtered.append(f)

            data.append({"city": city, "units": units, "list": lst_filtered})

        return data

    def render(self, table):
        # html = json2table.convert(table, build_direction="LEFT_TO_RIGHT", table_attributes={ "class" : "table table-sm caption-top meta-table table-responsive font-mono", "style" : "list-style-type: none !important;"})
        html = ""

        return (
            """<div id="meteogram"></div><div class="root" style="margin-top: 40px">"""
            + html
            + """</div>
        <script>
            if (typeof forecast === 'undefined') {
                var forecast = """
            + json.dumps(table)
            + """;
                // var tree = jsonview.create(forecast);
            } else {
                forecast = """
            + json.dumps(table)
            + """;
               // tree = jsonview.create(forecast);
            }
            
            // jsonview.render(tree, document.querySelector('.root'));
            // jsonview.expand(tree);
            window.meteogram = new Meteogram(forecast, 'meteogram');
        </script>"""
        )


class IceHockey(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, name="ice_hockey")

    def get_data(self, split):
        with open(f"{self.data_path}/{self.name}/{split}.json") as f:
            data = json.load(f)

        # recursively remove any references to images: `logo` and `flag` keys
        def recursive_remove_key(data, key_to_remove):
            if isinstance(data, dict):
                # Remove the key if it exists
                data.pop(key_to_remove, None)

                # Recursively call the function for nested dictionaries
                for key, value in data.items():
                    data[key] = recursive_remove_key(value, key_to_remove)

            elif isinstance(data, list):
                # Recursively call the function for elements in the list
                data = [recursive_remove_key(item, key_to_remove) for item in data]

            return data

        for game in data:
            start_timestamp = game["startTimestamp"]

            for key in [
                "changes",
                "crowdsourcingDataDisplayEnabled",
                "crowdsourcingEnabled",
                "customId",
                "finalResultOnly",
                "hasEventPlayerStatistics",
                "hasGlobalHighlights",
                "isEditor",
                "periods",
                "status",
                "time",
                "roundInfo",
                "tournament",
                "winnerCode",
            ]:
                if key in game:
                    game.pop(key)

            game["season"].pop("editor", None)

            for key in [
                "current",
                "slug",
                "sport",
                "teamColors",
                "subTeams",
                "userCount",
                "type",
                "disabled",
                "national",
            ]:
                recursive_remove_key(game, key)

            for key in ["homeTeam", "awayTeam"]:
                if (
                    type(game[key]["country"]) is dict
                    and "name" in game[key]["country"]
                ):
                    country_name = game[key]["country"]["name"]
                    game[key].pop("country")
                    game[key]["country"] = country_name

            # convert timestamp to date
            game["startDatetime"] = datetime.fromtimestamp(start_timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        return data

    def render(self, table):
        # metadata table
        metadata_columns = [
            "id",
            "startDatetime",
            "startTimestamp",
        ]
        metadata_trs = []

        home_team = table["homeTeam"]["name"]
        away_team = table["awayTeam"]["name"]
        match_info = f"{home_team} – {away_team}"

        for col in metadata_columns:
            th_name = h("th")(col)
            td_val = h("td")(table[col])
            tr = h("tr")(th_name, td_val)
            metadata_trs.append(tr)

        metadata_table_el = h(
            "table",
            klass="table table-sm table-bordered caption-top meta-table font-mono",
        )(h("caption")("metadata"), h("tbody")(metadata_trs))

        simple_table_names = [
            "season",
            "homeTeam",
            "homeScore",
            "awayScore",
            "awayTeam",
        ]
        simple_tables = []

        for table_name in simple_table_names:
            simple_trs = []
            for name, value in table[table_name].items():
                th_name = h("th")(name)
                td_val = h("td")(value)
                tr = h("tr")(th_name, td_val)
                simple_trs.append(tr)

            simple_table_el = h(
                "table",
                klass="table table-sm table-bordered caption-top meta-table font-mono",
            )(h("caption")(table_name), h("tbody")(simple_trs))
            simple_tables.append(simple_table_el)

        header_el = h("div")(h("h4", klass="")(match_info))
        col_1 = h("div", klass="col-5 ")(metadata_table_el, *simple_tables[:-2])
        col_2 = h("div", klass="col-7 ")(*simple_tables[-2:])
        cols = h("div", klass="row")(col_1, col_2)
        html_el = h("div")(header_el, cols)

        return html_el.render()


class GSMArena(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, name="gsmarena")

    def render(self, table):
        details = table["details"]

        quick_trs = []

        for spec in details["quickSpec"]:
            th_name = h("th")(spec["name"])
            td_val = h("td")(spec["value"])
            tr = h("tr")(th_name, td_val)
            quick_trs.append(tr)

        trs = []
        for category in details["detailSpec"]:
            category_name = category["category"]
            specs = category["specifications"]
            th_el = h("th", rowspan=len(specs) + 1)(category_name)
            tds = [th_el]

            for spec in specs:
                th_name = h("th")(spec["name"])
                td_val = h("td")(spec["value"])
                tr = h("tr")(th_name, td_val)
                tds.append(tr)

            tr_el = h("tr")(tds)
            trs.append(tr_el)

        product_info = "name: " + table["name"] + ", id: " + table["id"]

        quick_tbody_el = h("tbody")(quick_trs)
        quick_table_el = h(
            "table",
            klass="table table-sm table-bordered caption-top subtable font-mono",
        )(h("caption")("quick specifications"), h("tbody")(quick_tbody_el))

        tbody_el = h("tbody", id="main-table-body")(trs)
        table_el = h(
            "table",
            klass="table table-sm table-bordered caption-top main-table font-mono",
        )(h("caption")("detailed specifications"), tbody_el)

        header_el = h("div")(
            h("h4", klass="")(details["name"]), h("p", klass="")(product_info)
        )
        html_el = h("div")(header_el, quick_table_el, table_el)

        return html_el.render()


class OurWorldInData(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, name="owid")

    def get_data(self, split):
        data = []
        split_dir = Path(f"{self.data_path}/{self.name}/{split}")
        filenames = sorted(split_dir.iterdir(), key=lambda x: int(x.stem.split("-")[0]))

        for filename in filenames:
            with open(filename) as f:
                data.append(f.read())

        return data

    def render(self, table):
        # parse the csv comments, e.g. 'country: country_name' as Python dict
        lines_starting_with_hash = [
            line[1:].strip().split(": ")
            for line in table.split("\n")
            if line.startswith("#")
        ]

        metadata = {k: v for k, v in lines_starting_with_hash}

        title = metadata["title"]
        description = metadata["description"]
        country = metadata["country"]
        unit = metadata["unit"]
        data = []
        csv_lines = [line for line in table.split("\n") if not line.startswith("#")]
        # unit = csv_lines[0].split(",")[1]

        for row in csv_lines:
            if not row or len(row.split(",")) != 2 or "date" in row:
                continue
            date, value = row.split(",")
            # convert date to timestamp
            date = dateutil.parser.parse(date)
            date = date.timestamp() * 1000

            data.append([int(date), float(value)])

        # data per year vs. data per day
        date_format = (
            "%Y"
            if title in ["Deaths in under-fives", "Life expectancy at birth"]
            else "%Y-%m-%d"
        )

        return (
            """
        <div id="chartPlaceholder"></div>
        <script>
        if (typeof chartData === 'undefined') {
            var chartData = """
            + json.dumps(data)
            + """;
        } else {
            chartData = """
            + json.dumps(data)
            + """;
        }
        Highcharts.chart('chartPlaceholder', {
            chart: {
                zooming : {
                    enabled: false
                },
                animation: false,
                credits: {
                    enabled: false
                }
            },
            credits: {
                enabled: false
            },
            title: {
                text: '"""
            + f"{country}"
            + """',
                align: 'left'
            },
            subtitle: {
                text: '"""
            + f"{title}. {description}"
            + """',
                align: 'left'
            },
            xAxis: {
                type: 'datetime'
            },
            yAxis: {
                title: {
                    text: '"""
            + unit
            + """'
                }
            },
            legend: {
                enabled: false
            },
            plotOptions: {
                area: {
                    color: '#a6a6a6',
                    fillColor: '#f2f2f2',
                    marker: {
                        radius: 2,
                        fillColor: '#a6a6a6'
                    },
                    lineWidth: 1,
                    tooltip: {
                        dateTimeLabelFormats: {
                            hour: '"""
            + date_format
            + """',
                        }
                    },
                    states: {
                        hover: {
                            lineWidth: 1
                        }
                    },
                    threshold: null
                }
            },

            series: [{
                type: 'area',
                name: '"""
            + country
            + """',
                data: chartData,
                animation: false
            }]
        });
        </script>
        """
        )


class Wikidata(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, name="wikidata")

    def get_data(self, split):
        with open(f"{self.data_path}/{self.name}/{split}.json") as f:
            data = json.load(f)

        examples = []

        for example in data:
            entity = example["entity"]
            properties = example["properties"]

            table = entity + "\n---\n"
            table += "\n".join([f"- {prop}: {subj}" for prop, subj in properties])
            examples.append(table)

        return examples

    def render(self, table):
        table = table.split("\n")
        title = table[0]

        trs = []
        for line in table[2:]:
            key, value = line.split(": ", 1)
            key = key.strip("- ")
            th_el = h("th")(key)
            td_el = h("td")(value)
            tr_el = h("tr")(th_el, td_el)
            trs.append(tr_el)

        tbody_el = h("tbody", id="main-table-body")(trs)
        table_el = h(
            "table",
            klass="table table-sm table-bordered caption-top main-table font-mono",
        )(tbody_el)

        header_el = h("div")(h("h4", klass="")(title))
        html_el = h("div")(header_el, table_el)

        return html_el.render()


if __name__ == "__main__":
    random.seed(42)
