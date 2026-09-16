import csv
import hashlib
import io
import json
from pathlib import Path
import pytest
from examples.food_reviews import parse_csv, download, operations


def fixture_bytes(vector="[0.6, 0.8]", rows=1):
    output=io.StringIO()
    writer=csv.writer(output)
    writer.writerow(["", "ProductId", "Score", "Summary", "Text", "combined", "embedding"])
    for index in range(rows):
        writer.writerow([str(index),"product-1",5,"Good","Good food","Title: Good; Content: Good food",vector])
    return output.getvalue().encode()


def lock(data,rows=1):
    return {"sha256":hashlib.sha256(data).hexdigest(),"dimensions":2,"expected_rows":rows,
            "embedding_model":"unspecified","url":"https://example.invalid/data.csv","bytes":len(data)}


def test_validated_vectors_and_stable_ids():
    data=fixture_bytes()
    a=parse_csv(data,lock(data));b=parse_csv(data,lock(data))
    assert a==b and a[0]["vector"]==[.6,.8]
    assert a[0]["payload"]["score"]==5
    assert a[0]["payload"]["review_id"]=="0"
    assert a[0]["payload"]["embedding_provider"]=="OpenAI"
    assert [o["type"] for o in operations(8)]==["cluster","topics","sentiment","outliers"]


@pytest.mark.parametrize("vector",["[1]","[True, 1]","[0,0]","[1, float('nan')]","__import__('os').system('false')"])
def test_invalid_vectors_rejected_without_execution(vector):
    data=fixture_bytes(vector)
    with pytest.raises((ValueError,SyntaxError)):
        parse_csv(data,lock(data))


def test_verified_cache_does_not_download(tmp_path,monkeypatch):
    data=fixture_bytes()
    (tmp_path/"food_reviews.csv").write_bytes(data)
    def forbidden(*args,**kwargs):
        raise AssertionError("Should reuse the checksum-verified cache")
    monkeypatch.setattr("urllib.request.urlopen",forbidden)
    assert download(lock(data),tmp_path).read_bytes()==data


def test_changed_download_rejected(tmp_path,monkeypatch):
    data=fixture_bytes()
    monkeypatch.setattr("urllib.request.urlopen",lambda *a,**k:io.BytesIO(b"unexpected bytes"))
    with pytest.raises(ValueError,match="checksum/size"):
        download(lock(data),tmp_path)
    assert not (tmp_path/"food_reviews.csv").exists()
    assert not list(tmp_path.glob("*.tmp"))
