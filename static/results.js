const Result = props =>{
const item = props.result;

return (
    <div class='rel_result'>
        <div class='rel_score'> {item.score.toFixed(2)} </div>
        <div class='rel_title'> {item.title} </div>
        <div class='rel_content'> {item.context} </div>
    </div>
    )
}

const QAResult = props =>{
const item = props.qa_result;

return (
    <div class='rel_qa_result'>
        <div class='rel_score'> Confidence: {item.score.toFixed(2)} </div>
        <div class='rel_title'> Answer: {item.answer} </div>
        <div class='rel_content'> {item.context} </div>
    </div>
    )
}

const LFQAResult = props =>{
const item = props.lfqa_result;

return (
    <div class='rel_lfqa_result'>
        <div class='rel_content'> {item.gen_content} </div>
    </div>
    )
}

const ResultsList = props => {
    const lst = props.results;

    const rlst = lst.map((result, idx) => <Result key={idx} result={result} />);
    return (
        <div>
          <div id="resultList" class="rel_results">
            <div class="rel_content"></div>
            {rlst}
          </div>
        </div>
    )
}

const QAList = props => {
    const lst = props.qa_results;

    const rlst = lst.map((qa_result, idx) => <QAResult key={idx} qa_result={qa_result} />);
    return (
        <div>
          <div id="resultList" class="rel_results">
            <div class="rel_content"></div>
            {rlst}
          </div>
        </div>
    )
}

ReactDOM.render(<ResultsList results={results} />, document.getElementById('search-result') );
ReactDOM.render(<QAList qa_results={qa_results} />, document.getElementById('factoidqa') );
ReactDOM.render(<LFQAResult lfqa_result={lfqa_result} />, document.getElementById('lfqa') );