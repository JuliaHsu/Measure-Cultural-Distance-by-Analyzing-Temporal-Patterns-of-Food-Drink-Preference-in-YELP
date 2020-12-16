
var styleJSON = {
    "version": 8,
    "name": "qgis2web export",
    "pitch": 0,
    "light": {
        "intensity": 0.2
    },
    "sources": {
        "pittsburgh_0": {
            "type": "geojson",
            "data": json_pittsburgh_0
        }
                    ,
        "PittsburghNeighCluster_1": {
            "type": "geojson",
            "data": json_PittsburghNeighCluster_1
        }
                    ,
        "Joinedlayer_2": {
            "type": "geojson",
            "data": json_Joinedlayer_2
        }
                    },
    "sprite": "",
    "glyphs": "https://glfonts.lukasmartinelli.ch/fonts/{fontstack}/{range}.pbf",
    "layers": [
        {
            "id": "background",
            "type": "background",
            "layout": {},
            "paint": {
                "background-color": "#ffffff"
            }
        },
        {
            "id": "lyr_pittsburgh_0_0",
            "type": "fill",
            "source": "pittsburgh_0",
            "layout": {},
            "paint": {'fill-opacity': 1.0, 'fill-color': '#729b6f'}
        }
,
        {
            "id": "lyr_PittsburghNeighCluster_1_0",
            "type": "circle",
            "source": "PittsburghNeighCluster_1",
            "layout": {},
            "paint": {'circle-radius': ['/', 7.142857142857142, 2], 'circle-color': '#f3a6b2', 'circle-opacity': 1.0, 'circle-stroke-width': 1, 'circle-stroke-color': '#232323'}
        }
,
        {
            "id": "lyr_Joinedlayer_2_0",
            "type": "fill",
            "source": "Joinedlayer_2",
            "layout": {},
            "paint": {'fill-opacity': ['case', ['==', ['get', 'cluster'], 0], 1.0, ['==', ['get', 'cluster'], 1], 1.0, ['==', ['get', 'cluster'], 2], 1.0, 1.0], 'fill-color': ['case', ['==', ['get', 'cluster'], 0], '#440154', ['==', ['get', 'cluster'], 1], '#31688e', ['==', ['get', 'cluster'], 2], '#35b779', '#ffffff']}
        }
],
}