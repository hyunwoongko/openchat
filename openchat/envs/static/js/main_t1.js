function Message(arg) {
    this.text = arg.text;
    this.message_side = arg.message_side;

    this.draw = function (_this) {
        return function () {
            let $message;
            $message = $($('.message_template').clone().html());
            $message.addClass(_this.message_side).find('.text').html(_this.text);
            $('.messages').append($message);

            return setTimeout(function () {
                return $message.addClass('appeared');
            }, 0);
        };
    }(this);
    return this;
}

function getMessageText() {
    let $message_input;
    $message_input = $('.message_input');
    return $message_input.val();
}

function sendMessage(text, message_side) {
    let $messages, message;
    $('.message_input').val('');
    $messages = $('.messages');
    message = new Message({
        text: text,
        message_side: message_side
    });
    message.draw();
    $messages.animate({scrollTop: $messages.prop('scrollHeight')}, 300);
}

function onClickAsEnter(e) {
    if (e.keyCode === 13) {
        onSendButtonClicked()
    }
}

function greet() {
    setTimeout(function () {
        return sendMessage("Welcome to Open Chat's Demo !", 'left');
    }, 1000);

    setTimeout(function () {
        return sendMessage("Talk with AI easily using Open Chat !", 'left');
    }, 1500);

    setTimeout(function () {
        return sendMessage("Say hello to AI.", 'left');
    }, 2000);

    setTimeout(function () {
        return sendMessage('Type ".clear" if you want to clear the conversation.', 'left');
    }, 2500);
}


function requestChat(userId, botId, topic, messageText) {
    $.ajax({
        url: "/send/" + userId,
        type: "POST",
        dataType: "json",
        data: {"text": messageText,
               "bot_id": botId,
               "topic": topic},

        success: function (data) {
            console.log(data)
            return sendMessage(data["output"], 'left');
        },
        error: function (request, status, error) {
            console.log(error);
            return sendMessage('Communication failed.', 'left');
        },
        always: function(response) {
            console.log(response)
        }
    });
}

function onSendButtonClicked() {
    let messageText = getMessageText();
    let userId = document.getElementById("userId").value;
    let botId = document.getElementById("botId").value;
    let topic = document.getElementById("topic").value;

    if (userId == '') {
        document.getElementById('warning').innerText = 'Please fill Nickname!';
        return ;
    }

    userId = btoa(userId);

    sendMessage(messageText, 'right');
    return requestChat(userId, botId, topic, messageText);
}
