import io,json
import numpy as np
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import NameObject,DictionaryObject,NumberObject,DecodedStreamObject,StreamObject
from ds41f_mlx.web_artifacts import ArtifactStore
from ds41f_mlx.web_binary_tools import FetchPDFTool
from ds41f_mlx.web_tools import AcquiredResource


def retain(tmp_path,data):
    store=ArtifactStore(tmp_path/'a.db')
    return store,store.put(AcquiredResource('https://example.com/a.pdf','https://example.com/a.pdf','application/pdf',data,False,0),'pdf')


def test_metadata_cannot_inject_arbitrary_info_keys(tmp_path):
    writer=PdfWriter();writer.add_blank_page(width=600,height=780)
    writer.add_metadata({'/'+'X'*100000:'hidden payload','/Title':'t'*2000})
    stream=io.BytesIO();writer.write(stream)
    store,artifact=retain(tmp_path,stream.getvalue())
    result=FetchPDFTool(store).run({'artifact_id':artifact.id,'mode':'text'})
    value=json.loads(result.content)
    assert value['metadata_truncated'] and len(value['metadata']['Title'])==1000
    assert 'hidden payload' not in result.content and len(result.content)<10000


def test_native_raster_output_ceiling_has_resource_classification(tmp_path):
    data=np.random.default_rng(0).integers(0,256,(2048,2048,3),dtype=np.uint8)
    stream=io.BytesIO();Image.fromarray(data).save(stream,format='JPEG',quality=75)
    writer=PdfWriter();obj=StreamObject();obj._data=stream.getvalue()
    obj.update({NameObject('/Type'):NameObject('/XObject'),NameObject('/Subtype'):NameObject('/Image'),NameObject('/Filter'):NameObject('/DCTDecode'),NameObject('/Width'):NumberObject(2048),NameObject('/Height'):NumberObject(2048),NameObject('/ColorSpace'):NameObject('/DeviceRGB'),NameObject('/BitsPerComponent'):NumberObject(8)})
    image=writer._add_object(obj)
    for _ in range(4):
        page=writer.add_blank_page(width=1365,height=1365)
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/XObject'):DictionaryObject({NameObject('/I'):image})})
        commands=DecodedStreamObject();commands.set_data(b'q 1365 0 0 1365 0 0 cm /I Do Q')
        page[NameObject('/Contents')]=writer._add_object(commands)
    stream=io.BytesIO();writer.write(stream);original=stream.getvalue();assert len(original)<8*1024*1024
    store,artifact=retain(tmp_path,original)
    result=FetchPDFTool(store).run({'artifact_id':artifact.id,'pages':[1,2,3,4],'mode':'visual'})
    assert json.loads(result.content)['error_code']=='pdf_resource_ceiling'
    assert store.get(artifact.id,'pdf').resource.data==original
